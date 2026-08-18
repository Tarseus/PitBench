import os
import subprocess
import sys


def test_importing_claude_code_agents_ignores_invalid_unused_proxy() -> None:
    env = os.environ | {"ALL_PROXY": "socks://127.0.0.1:7897"}
    command = [
        sys.executable,
        "-c",
        "from fceval.agents.agent_factory import AgentFactory; "
        "from fceval.agents.agent_name import AgentName; "
        "assert AgentFactory.get_agent_class(AgentName.CLAUDE_CODE).name() "
        "== 'claude-code'",
    ]

    result = subprocess.run(command, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
