import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def test_run_agent_importable():
    spec = importlib.util.spec_from_file_location(
        "run_agent",
        ROOT / "run_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_agent_loop_importable():
    from agent.loop import AgentLoop
    assert AgentLoop is not None


def test_data_and_outputs_dirs_exist():
    assert (ROOT / "data").is_dir()
    assert (ROOT / "outputs").is_dir()


def test_env_example_has_api_key():
    content = (ROOT / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY" in content
