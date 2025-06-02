"""Tests for HED service integration module."""

import pytest
import time
from unittest.mock import MagicMock, patch
from aioresponses import aioresponses

from hed_tools.hed_integration.service_integration import (
    DirectAPIIntegration,
    WebServiceIntegration,
    HEDIntegrationManager,
    IntegrationMethod,
    IntegrationResult,
    create_hed_integration_manager,
)


class TestDirectAPIIntegration:
    """Test direct Python API integration."""

    @pytest.fixture
    def integration(self):
        """Create DirectAPIIntegration instance."""
        config = {"dataset_name": "Test Dataset"}
        return DirectAPIIntegration(config)

    @pytest.mark.asyncio
    async def test_is_available_with_hed_library(self, integration):
        """Test availability check when HED library is available."""
        with patch("hed_tools.hed_integration.service_integration.HED_AVAILABLE", True):
            with patch.object(
                integration, "load_schema", return_value=IntegrationResult(success=True)
            ):
                available = await integration.is_available()
                assert available is True

    @pytest.mark.asyncio
    async def test_is_available_without_hed_library(self, integration):
        """Test availability check when HED library is not available."""
        with patch(
            "hed_tools.hed_integration.service_integration.HED_AVAILABLE", False
        ):
            available = await integration.is_available()
            assert available is False

    @pytest.mark.asyncio
    async def test_validate_hed_string_success(self, integration):
        """Test successful HED string validation."""
        mock_schema = MagicMock()
        integration.loaded_schemas["8.3.0"] = mock_schema

        with patch("hed_tools.hed_integration.service_integration.HED_AVAILABLE", True):
            with patch("hed.models.hed_string.HedString"):
                with patch(
                    "hed.validator.hed_validator.HedValidator"
                ) as mock_validator:
                    mock_validator_instance = mock_validator.return_value
                    mock_validator_instance.validate.return_value = []  # No issues

                    result = await integration.validate_hed_string(
                        "Event/Test", "8.3.0"
                    )

                    assert result.success is True
                    assert result.data["valid"] is True
                    assert result.data["hed_string"] == "Event/Test"
                    assert result.method_used == IntegrationMethod.DIRECT_API

    @pytest.mark.asyncio
    async def test_validate_hed_string_with_errors(self, integration):
        """Test HED string validation with validation errors."""
        mock_schema = MagicMock()
        integration.loaded_schemas["8.3.0"] = mock_schema

        with patch("hed_tools.hed_integration.service_integration.HED_AVAILABLE", True):
            with patch("hed.models.hed_string.HedString"):
                with patch(
                    "hed.validator.hed_validator.HedValidator"
                ) as mock_validator:
                    mock_validator_instance = mock_validator.return_value
                    mock_error = MagicMock()
                    mock_error.__str__ = lambda: "Invalid tag"
                    mock_validator_instance.validate.return_value = [mock_error]

                    result = await integration.validate_hed_string(
                        "InvalidTag", "8.3.0"
                    )

                    assert result.success is False
                    assert result.data["valid"] is False
                    assert "Invalid tag" in result.data["issues"]

    @pytest.mark.asyncio
    async def test_generate_sidecar_template(self, integration):
        """Test sidecar template generation."""
        with patch("hed_tools.hed_integration.service_integration.HED_AVAILABLE", True):
            with patch(
                "hed.tools.analysis.tabular_summary.TabularSummary"
            ) as mock_summary_class:
                mock_summary = mock_summary_class.return_value
                mock_summary.extract_sidecar_template.return_value = {
                    "test": "template"
                }

                result = await integration.generate_sidecar_template(
                    event_files=["test.tsv"],
                    value_columns=["rt"],
                    skip_columns=["onset"],
                )

                assert result.success is True
                assert result.data["template"] == {"test": "template"}
                assert result.method_used == IntegrationMethod.DIRECT_API

    @pytest.mark.asyncio
    async def test_load_schema_caching(self, integration):
        """Test schema loading and caching."""
        with patch("hed_tools.hed_integration.service_integration.HED_AVAILABLE", True):
            with patch(
                "hed_tools.hed_integration.service_integration.hed_schema.load_schema"
            ) as mock_load:
                mock_schema = MagicMock()
                mock_schema.version_number = "8.3.0"
                mock_schema.tags = {"Event": {}, "Action": {}}
                mock_load.return_value = mock_schema

                # First load
                result1 = await integration.load_schema("8.3.0")
                assert result1.success is True
                assert result1.data["cached"] is False

                # Second load should be cached
                result2 = await integration.load_schema("8.3.0")
                assert result2.success is True
                assert result2.data["cached"] is True

                # Should only call load_schema once
                mock_load.assert_called_once()


class TestWebServiceIntegration:
    """Test web service integration."""

    @pytest.fixture
    def integration(self):
        """Create WebServiceIntegration instance."""
        config = {
            "base_url": "https://test-hedtools.example.com/hed",
            "timeout": 10.0,
            "max_retries": 2,
        }
        return WebServiceIntegration(config)

    @pytest.mark.asyncio
    async def test_is_available_service_up(self, integration):
        """Test availability check when service is up."""
        with aioresponses() as m:
            m.get("https://test-hedtools.example.com/hed/heartbeat", status=200)

            available = await integration.is_available()
            assert available is True

    @pytest.mark.asyncio
    async def test_is_available_service_down(self, integration):
        """Test availability check when service is down."""
        with aioresponses() as m:
            m.get("https://test-hedtools.example.com/hed/heartbeat", status=500)
            m.get("https://test-hedtools.example.com/hed/", status=500)

            available = await integration.is_available()
            assert available is False

    @pytest.mark.asyncio
    async def test_validate_hed_string_success(self, integration):
        """Test successful HED validation via web service."""
        response_data = {
            "valid": True,
            "issues": [],
            "hed_string": "Event/Test",
            "schema_version": "8.3.0",
        }

        with aioresponses() as m:
            m.post(
                "https://test-hedtools.example.com/hed/validate",
                payload=response_data,
                status=200,
            )

            result = await integration.validate_hed_string("Event/Test", "8.3.0")

            assert result.success is True
            assert result.data == response_data
            assert result.method_used == IntegrationMethod.WEB_SERVICE

    @pytest.mark.asyncio
    async def test_validate_hed_string_with_retries(self, integration):
        """Test HED validation with retry logic."""
        response_data = {"valid": True, "issues": []}

        with aioresponses() as m:
            # First attempt fails, second succeeds
            m.post("https://test-hedtools.example.com/hed/validate", status=500)
            m.post(
                "https://test-hedtools.example.com/hed/validate",
                payload=response_data,
                status=200,
            )

            result = await integration.validate_hed_string("Event/Test")

            assert result.success is True

    @pytest.mark.asyncio
    async def test_generate_sidecar_template_web(self, integration):
        """Test sidecar template generation via web service."""
        response_data = {
            "template": {
                "trial_type": {"HED": {"go": "Event/Go", "stop": "Event/Stop"}}
            },
            "summary": {"columns": 2},
        }

        with aioresponses() as m:
            m.post(
                "https://test-hedtools.example.com/hed/generate_sidecar",
                payload=response_data,
                status=200,
            )

            result = await integration.generate_sidecar_template(
                event_files=["test.tsv"], value_columns=["trial_type"]
            )

            assert result.success is True
            assert result.data == response_data

    @pytest.mark.asyncio
    async def test_load_schema_web(self, integration):
        """Test schema loading via web service."""
        schema_data = {
            "version": "8.3.0",
            "tags": {"Event": {"description": "Event tag"}},
        }

        with aioresponses() as m:
            m.get(
                "https://test-hedtools.example.com/hed/schema/8.3.0",
                payload=schema_data,
                status=200,
            )

            result = await integration.load_schema("8.3.0")

            assert result.success is True
            assert result.data == schema_data

    @pytest.mark.asyncio
    async def test_session_management(self, integration):
        """Test HTTP session management."""
        # Session should be created on first use
        assert integration.session is None

        with aioresponses() as m:
            m.get("https://test-hedtools.example.com/hed/heartbeat", status=200)

            await integration.is_available()
            assert integration.session is not None

            # Clean up
            await integration.close()


class TestHEDIntegrationManager:
    """Test HED integration manager."""

    @pytest.fixture
    def manager_config(self):
        """Configuration for integration manager."""
        return {
            "preferred_method": "auto_select",
            "enable_performance_comparison": True,
            "direct_api": {"dataset_name": "Test Dataset"},
            "web_service": {
                "base_url": "https://test-hedtools.example.com/hed",
                "timeout": 10.0,
            },
        }

    @pytest.fixture
    def manager(self, manager_config):
        """Create HEDIntegrationManager instance."""
        return HEDIntegrationManager(manager_config)

    @pytest.mark.asyncio
    async def test_auto_select_direct_api(self, manager):
        """Test auto-selection when only direct API is available."""
        with patch.object(
            manager.integrations[IntegrationMethod.DIRECT_API],
            "is_available",
            return_value=True,
        ):
            with patch.object(
                manager.integrations[IntegrationMethod.WEB_SERVICE],
                "is_available",
                return_value=False,
            ):
                method = await manager._select_integration_method()
                assert method == IntegrationMethod.DIRECT_API

    @pytest.mark.asyncio
    async def test_auto_select_web_service(self, manager):
        """Test auto-selection when only web service is available."""
        with patch.object(
            manager.integrations[IntegrationMethod.DIRECT_API],
            "is_available",
            return_value=False,
        ):
            with patch.object(
                manager.integrations[IntegrationMethod.WEB_SERVICE],
                "is_available",
                return_value=True,
            ):
                method = await manager._select_integration_method()
                assert method == IntegrationMethod.WEB_SERVICE

    @pytest.mark.asyncio
    async def test_auto_select_performance_based(self, manager):
        """Test auto-selection based on performance metrics."""
        # Set up different performance metrics
        manager.integrations[IntegrationMethod.DIRECT_API].metrics.average_latency = 0.5
        manager.integrations[
            IntegrationMethod.WEB_SERVICE
        ].metrics.average_latency = 1.0

        with patch.object(
            manager.integrations[IntegrationMethod.DIRECT_API],
            "is_available",
            return_value=True,
        ):
            with patch.object(
                manager.integrations[IntegrationMethod.WEB_SERVICE],
                "is_available",
                return_value=True,
            ):
                method = await manager._select_integration_method()
                assert method == IntegrationMethod.DIRECT_API  # Faster method

    @pytest.mark.asyncio
    async def test_validate_hed_string_manager(self, manager):
        """Test HED validation through manager."""
        mock_result = IntegrationResult(
            success=True, data={"valid": True}, method_used=IntegrationMethod.DIRECT_API
        )

        with patch.object(
            manager,
            "_select_integration_method",
            return_value=IntegrationMethod.DIRECT_API,
        ):
            with patch.object(
                manager.integrations[IntegrationMethod.DIRECT_API],
                "validate_hed_string",
                return_value=mock_result,
            ):
                result = await manager.validate_hed_string("Event/Test")

                assert result.success is True
                assert result.method_used == IntegrationMethod.DIRECT_API

    @pytest.mark.asyncio
    async def test_no_methods_available(self, manager):
        """Test behavior when no integration methods are available."""
        with patch.object(manager, "_select_integration_method", return_value=None):
            result = await manager.validate_hed_string("Event/Test")

            assert result.success is False
            assert "No integration methods available" in result.error

    @pytest.mark.asyncio
    async def test_performance_comparison(self, manager):
        """Test performance comparison functionality."""
        manager.comparison_enabled = True

        # Mock both integrations as available
        with patch.object(
            manager.integrations[IntegrationMethod.DIRECT_API],
            "is_available",
            return_value=True,
        ):
            with patch.object(
                manager.integrations[IntegrationMethod.WEB_SERVICE],
                "is_available",
                return_value=True,
            ):
                # Mock validation results
                direct_result = IntegrationResult(success=True, data={"valid": True})
                web_result = IntegrationResult(success=True, data={"valid": True})

                with patch.object(
                    manager.integrations[IntegrationMethod.DIRECT_API],
                    "validate_hed_string",
                    return_value=direct_result,
                ):
                    with patch.object(
                        manager.integrations[IntegrationMethod.WEB_SERVICE],
                        "validate_hed_string",
                        return_value=web_result,
                    ):
                        await manager._perform_comparison(
                            "validate_hed_string", "Event/Test"
                        )

                        # Should have recorded comparison
                        assert len(manager.performance_history) > 0
                        assert (
                            manager.performance_history[-1]["operation"]
                            == "validate_hed_string"
                        )

    def test_get_performance_report(self, manager):
        """Test performance report generation."""
        # Add some mock performance history
        manager.performance_history = [
            {
                "timestamp": time.time(),
                "operation": "validate_hed_string",
                "results": {
                    IntegrationMethod.DIRECT_API: {
                        "success": True,
                        "execution_time": 0.5,
                        "data_size": 100,
                    },
                    IntegrationMethod.WEB_SERVICE: {
                        "success": True,
                        "execution_time": 1.0,
                        "data_size": 150,
                    },
                },
            }
        ]

        report = manager.get_performance_report()

        assert "integration_methods" in report
        assert "performance_comparison" in report
        assert "recommendations" in report

        # Should have comparison data
        if "validate_hed_string" in report["performance_comparison"]:
            comparison = report["performance_comparison"]["validate_hed_string"]
            assert comparison["faster_method"] == "direct_api"
            assert comparison["speedup_ratio"] == 2.0

    @pytest.mark.asyncio
    async def test_availability_caching(self, manager):
        """Test availability result caching."""
        # Mock availability check
        mock_integration = manager.integrations[IntegrationMethod.DIRECT_API]

        with patch.object(
            mock_integration, "is_available", return_value=True
        ) as mock_check:
            # First check
            available1 = await manager._check_availability(IntegrationMethod.DIRECT_API)
            assert available1 is True

            # Second check should use cache
            available2 = await manager._check_availability(IntegrationMethod.DIRECT_API)
            assert available2 is True

            # Should only call is_available once due to caching
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_integrations(self, manager):
        """Test closing all integrations."""
        with patch.object(
            manager.integrations[IntegrationMethod.WEB_SERVICE], "close"
        ) as mock_close:
            await manager.close()
            mock_close.assert_called_once()


class TestIntegrationFactory:
    """Test integration factory function."""

    def test_create_with_default_config(self):
        """Test creating manager with default configuration."""
        manager = create_hed_integration_manager()

        assert isinstance(manager, HEDIntegrationManager)
        assert manager.preferred_method == IntegrationMethod.AUTO_SELECT
        assert manager.comparison_enabled is False

    def test_create_with_custom_config(self):
        """Test creating manager with custom configuration."""
        config = {
            "preferred_method": "direct_api",
            "enable_performance_comparison": True,
            "direct_api": {"dataset_name": "Custom Dataset"},
        }

        manager = create_hed_integration_manager(config)

        assert manager.preferred_method == IntegrationMethod.DIRECT_API
        assert manager.comparison_enabled is True
        assert (
            manager.integrations[IntegrationMethod.DIRECT_API].config["dataset_name"]
            == "Custom Dataset"
        )

    def test_config_deep_merge(self):
        """Test deep merging of configuration dictionaries."""
        config = {
            "web_service": {
                "timeout": 60.0,  # Override default
                "custom_header": "test",  # Add new field
            }
        }

        manager = create_hed_integration_manager(config)
        web_config = manager.integrations[IntegrationMethod.WEB_SERVICE].config

        assert web_config["timeout"] == 60.0  # Overridden
        assert web_config["max_retries"] == 3  # Default preserved
        assert web_config["custom_header"] == "test"  # New field added


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_direct_api(self):
        """Test end-to-end workflow using direct API."""
        # Skip if not available
        try:
            import importlib.util

            if importlib.util.find_spec("hed") is None:
                pytest.skip("HED library not available")
        except ImportError:
            pytest.skip("HED library not available")

        manager = create_hed_integration_manager({"preferred_method": "direct_api"})

        # Test schema loading
        schema_result = await manager.load_schema("8.3.0")
        assert schema_result.success is True

        # Test HED validation
        validation_result = await manager.validate_hed_string("Event", "8.3.0")
        assert validation_result.success is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_fallback_behavior(self):
        """Test fallback behavior when preferred method fails."""
        config = {
            "preferred_method": "web_service",
            "web_service": {"base_url": "https://invalid-url.example.com"},
        }

        manager = create_hed_integration_manager(config)

        # Should fall back to direct API if available
        with patch.object(
            manager.integrations[IntegrationMethod.DIRECT_API],
            "is_available",
            return_value=True,
        ):
            with patch.object(
                manager.integrations[IntegrationMethod.DIRECT_API],
                "validate_hed_string",
                return_value=IntegrationResult(success=True, data={"valid": True}),
            ):
                result = await manager.validate_hed_string("Event")
                assert result.success is True

        await manager.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
