import pytest

from simready_bench.captures import check_renderer_log


def test_shader_compilation_failure_cannot_be_hidden_by_successful_png_output():
    log = "Warning: in _ValidateCompilation -- Failed to compile shader (GEOMETRY_SHADER): 0:910(2): error: no function with name 'HdGet_diffuseColor_st_transform2d'"
    with pytest.raises(RuntimeError, match='shader compilation failed'):
        check_renderer_log(log)


def test_texture_resolution_warning_does_not_fabricate_a_shader_failure():
    check_renderer_log('Warning: Failed to resolve reference @texture.<UDIM>.png@')
