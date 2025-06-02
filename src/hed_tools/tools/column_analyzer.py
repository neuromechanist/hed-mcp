"""BIDS events file column analysis and HED annotation tools.

This module provides functionality to analyze BIDS events files, classify columns,
and prepare them for HED annotation through automated sidecar generation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BIDSColumnAnalyzer:
    """Analyzer for BIDS events file columns and HED requirements.
    
    This class provides methods to:
    - Load and validate BIDS events files
    - Analyze column types and content
    - Classify columns for HED annotation purposes
    - Generate column statistics and unique value summaries
    """
    
    def __init__(self):
        """Initialize the BIDS column analyzer."""
        self.required_columns = ['onset', 'duration']
        self.timing_columns = ['onset', 'duration', 'response_time', 'stim_duration']
        self.skip_columns = ['onset', 'duration', 'trial_number', 'run_number']
        self.hed_candidate_columns = []
        self._last_analysis = None
    
    async def analyze_events_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze BIDS events file structure and columns.
        
        Args:
            file_path: Path to BIDS events file (TSV or CSV)
            
        Returns:
            Comprehensive analysis of the events file including column classification
        """
        try:
            logger.info(f"Analyzing BIDS events file: {file_path}")
            
            # Load the events file
            df = await self._load_events_file(file_path)
            if df is None:
                return {"error": f"Failed to load events file: {file_path}"}
            
            # Perform column analysis
            analysis = {
                "file_path": str(file_path),
                "file_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                },
                "columns": {},
                "bids_compliance": {},
                "hed_candidates": [],
                "recommendations": []
            }
            
            # Analyze each column
            for column in df.columns:
                column_analysis = await self._analyze_column(df[column], column)
                analysis["columns"][column] = column_analysis
                
                # Classify for HED annotation
                if self._is_hed_candidate(column, column_analysis):
                    analysis["hed_candidates"].append({
                        "column": column,
                        "type": column_analysis["type"],
                        "unique_values": column_analysis["unique_values"],
                        "priority": self._get_hed_priority(column, column_analysis)
                    })
            
            # Check BIDS compliance
            analysis["bids_compliance"] = self._check_bids_compliance(df)
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            self._last_analysis = analysis
            logger.info(f"Analysis completed: {len(analysis['hed_candidates'])} HED candidates found")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Events file analysis failed: {e}")
            return {"error": str(e)}
    
    async def _load_events_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load events file with appropriate format detection.
        
        Args:
            file_path: Path to events file
            
        Returns:
            Loaded DataFrame or None if loading failed
        """
        try:
            if file_path.suffix.lower() == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                # Try TSV first, then CSV
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    df = pd.read_csv(file_path)
            
            logger.info(f"Loaded events file: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load events file {file_path}: {e}")
            return None
    
    async def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze a single column for type, content, and HED suitability.
        
        Args:
            series: Pandas Series to analyze
            column_name: Name of the column
            
        Returns:
            Dictionary with column analysis results
        """
        analysis = {
            "name": column_name,
            "type": self._determine_column_type(series),
            "null_count": series.isnull().sum(),
            "unique_count": series.nunique(),
            "unique_values": [],
            "statistics": {},
            "hed_suitable": False,
            "skip_reason": None
        }
        
        # Get unique values (limited for memory efficiency)
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 50:  # Limit to 50 unique values for display
            analysis["unique_values"] = sorted([str(v) for v in unique_vals])
        else:
            analysis["unique_values"] = sorted([str(v) for v in unique_vals[:50]])
            analysis["unique_values"].append(f"... and {len(unique_vals) - 50} more")
        
        # Type-specific analysis
        if analysis["type"] == "numeric":
            analysis["statistics"] = {
                "mean": float(series.mean()) if not series.empty else 0,
                "std": float(series.std()) if not series.empty else 0,
                "min": float(series.min()) if not series.empty else 0,
                "max": float(series.max()) if not series.empty else 0,
                "median": float(series.median()) if not series.empty else 0
            }
        elif analysis["type"] == "categorical":
            value_counts = series.value_counts()
            analysis["statistics"] = {
                "most_common": str(value_counts.index[0]) if not value_counts.empty else None,
                "most_common_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "distribution": dict(value_counts.head(10).to_dict())
            }
        
        # Determine HED suitability
        analysis["hed_suitable"] = self._is_column_hed_suitable(column_name, analysis)
        if not analysis["hed_suitable"]:
            analysis["skip_reason"] = self._get_skip_reason(column_name, analysis)
        
        return analysis
    
    def _determine_column_type(self, series: pd.Series) -> str:
        """Determine the semantic type of a column.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            Column type: 'numeric', 'categorical', 'datetime', 'text', or 'mixed'
        """
        # Remove null values for analysis
        clean_series = series.dropna()
        
        if clean_series.empty:
            return "empty"
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(clean_series):
            return "numeric"
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(clean_series):
            return "datetime"
        
        # Check if categorical (limited unique values)
        unique_ratio = clean_series.nunique() / len(clean_series)
        if unique_ratio < 0.5 and clean_series.nunique() <= 50:
            return "categorical"
        elif unique_ratio >= 0.5:
            return "text"
        else:
            return "mixed"
    
    def _is_column_hed_suitable(self, column_name: str, analysis: Dict[str, Any]) -> bool:
        """Determine if a column is suitable for HED annotation.
        
        Args:
            column_name: Name of the column
            analysis: Column analysis results
            
        Returns:
            True if column should be included in HED annotation
        """
        # Skip timing and administrative columns
        if column_name.lower() in [col.lower() for col in self.skip_columns]:
            return False
        
        # Skip numeric columns that look like measurements
        if (analysis["type"] == "numeric" and 
            column_name.lower() in ['response_time', 'reaction_time', 'accuracy', 'score']):
            return False
        
        # Prefer categorical columns with reasonable number of unique values
        if analysis["type"] == "categorical" and 2 <= analysis["unique_count"] <= 20:
            return True
        
        # Accept some numeric columns if they appear to be categorical
        if (analysis["type"] == "numeric" and 
            analysis["unique_count"] <= 10 and 
            analysis["unique_count"] >= 2):
            return True
        
        # Skip text columns with too many unique values
        if analysis["type"] == "text" and analysis["unique_count"] > 50:
            return False
        
        return analysis["type"] in ["categorical", "text"] and analysis["unique_count"] >= 2
    
    def _get_skip_reason(self, column_name: str, analysis: Dict[str, Any]) -> str:
        """Get reason why a column was skipped for HED annotation.
        
        Args:
            column_name: Name of the column
            analysis: Column analysis results
            
        Returns:
            Human-readable reason for skipping
        """
        if column_name.lower() in [col.lower() for col in self.skip_columns]:
            return f"Timing/administrative column: {column_name}"
        
        if analysis["type"] == "numeric" and analysis["unique_count"] > 10:
            return "Continuous numeric data"
        
        if analysis["unique_count"] < 2:
            return "Insufficient unique values"
        
        if analysis["unique_count"] > 50:
            return "Too many unique values"
        
        return "Not suitable for HED annotation"
    
    def _is_hed_candidate(self, column_name: str, analysis: Dict[str, Any]) -> bool:
        """Check if column is a good candidate for HED annotation."""
        return analysis["hed_suitable"]
    
    def _get_hed_priority(self, column_name: str, analysis: Dict[str, Any]) -> str:
        """Determine priority level for HED annotation.
        
        Args:
            column_name: Name of the column
            analysis: Column analysis results
            
        Returns:
            Priority level: 'high', 'medium', or 'low'
        """
        # High priority for common BIDS columns
        high_priority_columns = ['trial_type', 'condition', 'stimulus', 'response', 'task']
        if any(term in column_name.lower() for term in high_priority_columns):
            return "high"
        
        # Medium priority for categorical with reasonable unique values
        if analysis["type"] == "categorical" and 2 <= analysis["unique_count"] <= 10:
            return "medium"
        
        return "low"
    
    def _check_bids_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check BIDS compliance of the events file.
        
        Args:
            df: Events DataFrame
            
        Returns:
            BIDS compliance report
        """
        compliance = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "required_columns_present": True
        }
        
        # Check required columns
        missing_required = [col for col in self.required_columns if col not in df.columns]
        if missing_required:
            compliance["valid"] = False
            compliance["required_columns_present"] = False
            compliance["errors"].append(f"Missing required columns: {missing_required}")
        
        # Check column naming conventions
        for col in df.columns:
            if not col.islower() or ' ' in col:
                compliance["warnings"].append(f"Column '{col}' should be lowercase with no spaces")
        
        # Check onset column
        if 'onset' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['onset']):
                compliance["errors"].append("'onset' column must be numeric")
            elif df['onset'].isnull().any():
                compliance["errors"].append("'onset' column contains null values")
        
        # Check duration column
        if 'duration' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['duration']):
                compliance["errors"].append("'duration' column must be numeric")
        
        if compliance["errors"]:
            compliance["valid"] = False
        
        return compliance
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results.
        
        Args:
            analysis: Complete analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # HED annotation recommendations
        hed_candidates = analysis["hed_candidates"]
        if hed_candidates:
            high_priority = [c for c in hed_candidates if c["priority"] == "high"]
            if high_priority:
                recommendations.append(
                    f"Prioritize HED annotation for high-priority columns: "
                    f"{[c['column'] for c in high_priority]}"
                )
        
        # BIDS compliance recommendations
        if not analysis["bids_compliance"]["valid"]:
            recommendations.append("Address BIDS compliance issues before HED annotation")
        
        # Column-specific recommendations
        for col_name, col_info in analysis["columns"].items():
            if col_info["null_count"] > 0:
                recommendations.append(f"Consider handling {col_info['null_count']} null values in '{col_name}'")
        
        if not hed_candidates:
            recommendations.append("No suitable columns found for HED annotation - check column types and values")
        
        return recommendations
    
    async def suggest_hed_annotations(self, column_data: pd.Series, 
                                    column_name: str = None) -> List[Dict[str, str]]:
        """Suggest HED annotations for column values.
        
        Args:
            column_data: Column data to analyze
            column_name: Optional column name for context
            
        Returns:
            List of suggested HED annotations for unique values
        """
        try:
            suggestions = []
            unique_values = column_data.dropna().unique()
            
            for value in unique_values[:20]:  # Limit to 20 suggestions
                suggestion = {
                    "value": str(value),
                    "suggested_hed": self._suggest_hed_for_value(str(value), column_name),
                    "confidence": "low"  # TODO: Implement confidence scoring
                }
                suggestions.append(suggestion)
            
            logger.info(f"Generated {len(suggestions)} HED suggestions for column '{column_name}'")
            return suggestions
            
        except Exception as e:
            logger.error(f"HED suggestion generation failed: {e}")
            return []
    
    def _suggest_hed_for_value(self, value: str, column_name: str = None) -> str:
        """Suggest a HED annotation for a specific value.
        
        Args:
            value: Value to annotate
            column_name: Context column name
            
        Returns:
            Suggested HED annotation string
        """
        # TODO: Implement intelligent HED suggestion logic
        # This should use knowledge of common HED patterns and value types
        
        # Simple placeholder logic
        value_lower = value.lower()
        
        if column_name and 'trial_type' in column_name.lower():
            return f"Event/Category/{value}"
        elif column_name and 'response' in column_name.lower():
            if value_lower in ['left', 'right']:
                return f"Agent-action/Move/Translate/{value}"
            else:
                return f"Agent-action/{value}"
        elif value_lower in ['go', 'nogo', 'stop']:
            return f"Event/Instruction/{value}"
        else:
            return f"Event/Category/{value}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the last analysis performed.
        
        Returns:
            Summary dictionary or None if no analysis has been performed
        """
        if self._last_analysis is None:
            return {"error": "No analysis has been performed yet"}
        
        analysis = self._last_analysis
        return {
            "file_analyzed": analysis.get("file_path"),
            "total_columns": len(analysis.get("columns", {})),
            "hed_candidates": len(analysis.get("hed_candidates", [])),
            "bids_compliant": analysis.get("bids_compliance", {}).get("valid", False),
            "recommendations_count": len(analysis.get("recommendations", []))
        }


def create_column_analyzer() -> BIDSColumnAnalyzer:
    """Factory function to create a BIDS column analyzer.
    
    Returns:
        Initialized BIDSColumnAnalyzer instance
    """
    return BIDSColumnAnalyzer() 