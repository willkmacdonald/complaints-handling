"""Tests for IMDRF code database and utilities."""

import pytest

from src.imdrf import (
    IMDRFCode,
    get_all_codes,
    get_ancestors,
    get_children,
    get_code_by_id,
    search_codes,
    validate_code,
)
from src.imdrf.codes import (
    clear_cache,
    get_codes_by_type,
    get_full_path,
    get_top_level_codes,
)
from src.models.enums import IMDRFCodeType


@pytest.fixture(autouse=True)
def clear_code_cache() -> None:
    """Clear the code cache before each test."""
    clear_cache()


class TestLoadCodes:
    """Tests for loading IMDRF codes."""

    def test_get_all_codes_returns_codes(self) -> None:
        """Test that codes are loaded from JSON files."""
        codes = get_all_codes()
        assert len(codes) > 0

    def test_codes_have_both_types(self) -> None:
        """Test that both device and patient problem codes are loaded."""
        codes = get_all_codes()
        device_codes = [c for c in codes if c.code_type == IMDRFCodeType.DEVICE_PROBLEM]
        patient_codes = [
            c for c in codes if c.code_type == IMDRFCodeType.PATIENT_PROBLEM
        ]

        assert len(device_codes) > 0, "Should have device problem codes"
        assert len(patient_codes) > 0, "Should have patient problem codes"


class TestGetCodeById:
    """Tests for get_code_by_id function."""

    def test_get_existing_code(self) -> None:
        """Test retrieving an existing code."""
        code = get_code_by_id("A0601")
        assert code is not None
        assert code.code_id == "A0601"
        assert code.name == "Material Integrity Problem"
        assert code.code_type == IMDRFCodeType.DEVICE_PROBLEM

    def test_get_nonexistent_code(self) -> None:
        """Test retrieving a code that doesn't exist."""
        code = get_code_by_id("INVALID")
        assert code is None

    def test_get_patient_problem_code(self) -> None:
        """Test retrieving a patient problem code."""
        code = get_code_by_id("C01")
        assert code is not None
        assert code.name == "Death"
        assert code.code_type == IMDRFCodeType.PATIENT_PROBLEM


class TestValidateCode:
    """Tests for validate_code function."""

    def test_valid_code(self) -> None:
        """Test that valid codes return True."""
        assert validate_code("A0601") is True
        assert validate_code("C01") is True

    def test_invalid_code(self) -> None:
        """Test that invalid codes return False."""
        assert validate_code("INVALID") is False
        assert validate_code("") is False
        assert validate_code("A9999") is False


class TestGetChildren:
    """Tests for get_children function."""

    def test_get_children_of_parent(self) -> None:
        """Test getting children of a parent code."""
        children = get_children("A06")
        assert len(children) > 0

        child_ids = {c.code_id for c in children}
        assert "A0601" in child_ids
        assert "A0602" in child_ids

        # Verify all children have correct parent
        for child in children:
            assert child.parent_id == "A06"

    def test_get_children_of_leaf(self) -> None:
        """Test getting children of a leaf code (should be empty)."""
        children = get_children("A060102")  # Crack - leaf node
        assert children == []

    def test_get_children_of_nonexistent(self) -> None:
        """Test getting children of non-existent code."""
        children = get_children("INVALID")
        assert children == []


class TestGetAncestors:
    """Tests for get_ancestors function."""

    def test_get_ancestors_of_leaf(self) -> None:
        """Test getting ancestors of a deep leaf code."""
        ancestors = get_ancestors("A060102")  # Crack

        assert len(ancestors) == 2
        # Immediate parent first
        assert ancestors[0].code_id == "A0601"  # Material Integrity Problem
        assert ancestors[1].code_id == "A06"  # Material Problem

    def test_get_ancestors_of_top_level(self) -> None:
        """Test getting ancestors of top-level code (should be empty)."""
        ancestors = get_ancestors("A06")
        assert ancestors == []

    def test_get_ancestors_of_nonexistent(self) -> None:
        """Test getting ancestors of non-existent code."""
        ancestors = get_ancestors("INVALID")
        assert ancestors == []


class TestGetFullPath:
    """Tests for get_full_path function."""

    def test_full_path_of_leaf(self) -> None:
        """Test getting full path of a leaf code."""
        path = get_full_path("A060102")
        assert path == "Material Problem > Material Integrity Problem > Crack"

    def test_full_path_of_top_level(self) -> None:
        """Test getting full path of top-level code."""
        path = get_full_path("A06")
        assert path == "Material Problem"

    def test_full_path_of_nonexistent(self) -> None:
        """Test getting full path of non-existent code."""
        path = get_full_path("INVALID")
        assert path is None


class TestSearchCodes:
    """Tests for search_codes function."""

    def test_search_by_name(self) -> None:
        """Test searching by code name."""
        results = search_codes("material")
        assert len(results) > 0

        # Should find Material Problem
        names = [r.name for r in results]
        assert any("Material" in name for name in names)

    def test_search_by_description(self) -> None:
        """Test searching by description."""
        results = search_codes("crack")
        assert len(results) > 0

        # Should find codes related to cracking
        code_ids = {r.code_id for r in results}
        assert "A060102" in code_ids  # Crack

    def test_search_by_example(self) -> None:
        """Test searching by example text."""
        results = search_codes("pacemaker")
        assert len(results) > 0

    def test_search_with_type_filter(self) -> None:
        """Test searching with code type filter."""
        results = search_codes("problem", code_type=IMDRFCodeType.DEVICE_PROBLEM)

        for result in results:
            assert result.code_type == IMDRFCodeType.DEVICE_PROBLEM

    def test_search_limit(self) -> None:
        """Test that search respects limit parameter."""
        results = search_codes("a", limit=5)
        assert len(results) <= 5

    def test_search_no_results(self) -> None:
        """Test search with no matches."""
        results = search_codes("xyznonexistent123")
        assert results == []

    def test_exact_code_id_match_ranked_first(self) -> None:
        """Test that exact code ID match is ranked highest."""
        results = search_codes("A06")
        assert results[0].code_id == "A06"


class TestGetCodesByType:
    """Tests for get_codes_by_type function."""

    def test_get_device_problem_codes(self) -> None:
        """Test getting all device problem codes."""
        codes = get_codes_by_type(IMDRFCodeType.DEVICE_PROBLEM)
        assert len(codes) > 0

        for code in codes:
            assert code.code_type == IMDRFCodeType.DEVICE_PROBLEM

    def test_get_patient_problem_codes(self) -> None:
        """Test getting all patient problem codes."""
        codes = get_codes_by_type(IMDRFCodeType.PATIENT_PROBLEM)
        assert len(codes) > 0

        for code in codes:
            assert code.code_type == IMDRFCodeType.PATIENT_PROBLEM


class TestGetTopLevelCodes:
    """Tests for get_top_level_codes function."""

    def test_get_all_top_level(self) -> None:
        """Test getting all top-level codes."""
        codes = get_top_level_codes()
        assert len(codes) > 0

        for code in codes:
            assert code.is_top_level
            assert code.parent_id is None

    def test_get_top_level_by_type(self) -> None:
        """Test getting top-level codes filtered by type."""
        codes = get_top_level_codes(code_type=IMDRFCodeType.DEVICE_PROBLEM)

        for code in codes:
            assert code.is_top_level
            assert code.code_type == IMDRFCodeType.DEVICE_PROBLEM


class TestIMDRFCodeModel:
    """Tests for IMDRFCode model."""

    def test_code_properties(self) -> None:
        """Test IMDRFCode computed properties."""
        code = IMDRFCode(
            code_id="TEST01",
            name="Test Code",
            code_type=IMDRFCodeType.DEVICE_PROBLEM,
            parent_id=None,
            level=1,
        )
        assert code.is_top_level is True
        assert code.full_path == "Test Code"

    def test_child_code_properties(self) -> None:
        """Test IMDRFCode properties for child code."""
        code = IMDRFCode(
            code_id="TEST0101",
            name="Test Child",
            code_type=IMDRFCodeType.DEVICE_PROBLEM,
            parent_id="TEST01",
            level=2,
        )
        assert code.is_top_level is False

    def test_code_json_roundtrip(self) -> None:
        """Test JSON serialization of IMDRFCode."""
        code = IMDRFCode(
            code_id="A0601",
            name="Material Integrity Problem",
            code_type=IMDRFCodeType.DEVICE_PROBLEM,
            parent_id="A06",
            description="Material has lost structural integrity",
            examples=["Cracking", "Breaking"],
            level=2,
        )

        json_str = code.model_dump_json()
        loaded = IMDRFCode.model_validate_json(json_str)
        assert loaded == code


class TestCodeHierarchy:
    """Tests for code hierarchy navigation."""

    def test_navigate_hierarchy(self) -> None:
        """Test navigating up and down the hierarchy."""
        # Start at leaf
        leaf = get_code_by_id("A060102")  # Crack
        assert leaf is not None

        # Go up to parent
        ancestors = get_ancestors(leaf.code_id)
        parent = ancestors[0]
        assert parent.code_id == "A0601"

        # Go down from grandparent
        grandparent = ancestors[1]
        children = get_children(grandparent.code_id)
        child_ids = {c.code_id for c in children}
        assert parent.code_id in child_ids

    def test_all_children_have_valid_parents(self) -> None:
        """Test that all child codes reference valid parent codes."""
        all_codes = get_all_codes()

        for code in all_codes:
            if code.parent_id is not None:
                parent = get_code_by_id(code.parent_id)
                assert parent is not None, (
                    f"Code {code.code_id} has invalid parent {code.parent_id}"
                )
