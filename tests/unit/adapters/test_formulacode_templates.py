from adapters.formulacode.utils import (
    render_dockerfile,
    render_run_setup_sh,
    render_tests_sh,
)


def test_run_tests_restores_agent_patch_before_profiling() -> None:
    script = render_tests_sh(base_commit="deadbeef")

    reset_index = script.index("reset_repo_state deadbeef")
    apply_index = script.index('git apply --whitespace=nowarn "$PATCH_PATH"')
    profile_index = script.index('LOG_PATH="${LOG_DIR}/postrun_')

    assert reset_index < apply_index < profile_index


def test_runtime_scripts_do_not_install_evaluation_dependencies() -> None:
    dockerfile = render_dockerfile(base_image="example/image")
    test_script = render_tests_sh(base_commit="deadbeef")
    setup_script = render_run_setup_sh()

    assert "snapshot-tester" not in dockerfile
    assert "python -m pip install" not in dockerfile
    assert "snapshot-tester" not in test_script
    assert "pip install -U jinja2" not in test_script
    assert "uv pip install -q --upgrade coverage" not in setup_script
