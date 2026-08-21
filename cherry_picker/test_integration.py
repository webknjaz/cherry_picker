"""End-to-end tests exercising the cherry-picker CLI.

These tests drive the real CLI against real Git repositories, with
only the process boundaries — the GitHub HTTP API and the web
browser — replaced by test doubles.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from subprocess import check_output as ensure_cmd_succeeds
from typing import NamedTuple

import pytest
from pytest_mock import MockerFixture, MockType

from . import cherry_picker

TARGET_BACKPORT_BRANCH = "3.14"
EXPECTED_COAUTHOR_TRAILER = "Co-authored-by: Monty Python <bot@python.org>"

_parametrize_resolution_style = pytest.mark.parametrize(
    "resolve_with_cherry_pick_continue",
    (
        pytest.param(False, id="git-add-only"),
        pytest.param(True, id="git-cherry-pick-continue"),
    ),
)


@pytest.fixture(autouse=True, scope="session")
def _modern_git_preflight() -> None:
    """Skip the module when Git lacks ``git init --initial-branch``."""
    git_version_probe_cmd = "git", "--version"
    raw_git_version = ensure_cmd_succeeds(git_version_probe_cmd).decode("utf-8")
    git_version_match = re.search(r"(\d+)\.(\d+)", raw_git_version)
    assert git_version_match is not None
    git_version = tuple(int(number) for number in git_version_match.groups())
    if git_version < (2, 28):  # pragma: no cover
        pytest.skip("Git >= 2.28 is required for `git init --initial-branch`")


@pytest.fixture(autouse=True)
def isolated_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shield the tests from the host Git and GitHub configuration.

    CI runners have no Git identity configured, while development
    machines may have settings like ``commit.gpgsign`` or ``rerere``
    that would change the behavior of the Git commands under test.
    A real ``GH_AUTH`` token must also never leak into the test runs.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Monty Python")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "bot@python.org")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Monty Python")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "bot@python.org")
    monkeypatch.setenv("GIT_EDITOR", "true")
    monkeypatch.setenv("GH_AUTH", "sentinel-token-stub")


class CliResult(NamedTuple):
    """The subset of ``click.testing.Result`` these tests rely on."""

    exit_code: int
    output: str


@pytest.fixture
def invoke_cherry_picker(
    capsys: pytest.CaptureFixture[str],
) -> Callable[..., CliResult]:
    """Run ``cherry_pick_cli`` in-process and capture its result.

    ``cherry_pick_cli`` is a plain argparse-based function, not a Click
    command (Click was removed from this project's runtime dependencies),
    so it's called directly instead of through ``click.testing.CliRunner``.
    Every path under test here ends in either ``sys.exit()`` or falling off
    the end on success, so ``SystemExit`` is the only exception treated as
    a normal CLI exit -- anything else propagates and fails the test with
    its own traceback, same as an unhandled exception escaping the real
    console-script entry point would in production.
    """

    def _invoke(*argv: str) -> CliResult:
        capsys.readouterr()  # discard anything captured before this call
        try:
            cherry_picker.cherry_pick_cli(list(argv))
        except SystemExit as system_exit:
            exit_code = system_exit.code
            if exit_code is None:
                exit_code = 0
            elif not isinstance(exit_code, int):
                exit_code = 1
        else:
            exit_code = 0
        captured = capsys.readouterr()
        return CliResult(exit_code=exit_code, output=captured.out + captured.err)

    return _invoke


@pytest.fixture
def gh_create_pr_http_mock(mocker: MockerFixture) -> MockType:
    """Replace the GitHub PR creation HTTP call with a success stub.

    Mocking at the ``requests.post`` level keeps the real
    ``CherryPicker.create_gh_pr`` running, so the PR title and body
    extraction and the API payload stay under test. It also acts as
    a safety net against real network requests.
    """
    gh_pr_created_response_stub = mocker.Mock(status_code=HTTPStatus.CREATED)
    gh_pr_created_response_stub.json.return_value = {
        "html_url": "https://github.com/python/cpython/pull/1234",
        "number": 1234,
    }
    return mocker.patch.object(
        cherry_picker.requests,
        "post",
        return_value=gh_pr_created_response_stub,
    )


@pytest.fixture
def browser_mock(mocker: MockerFixture) -> MockType:
    """Prevent any code path from opening a real web browser."""
    return mocker.patch.object(cherry_picker.webbrowser, "open_new_tab")


@pytest.fixture
def upstream_remote_name() -> str:
    return "origin"


@pytest.fixture
def pr_remote_name() -> str:
    return "fork"


def _make_bare_repo(dot_git_path: Path) -> Path:
    git_init_bare_cmd = (
        "git",
        "init",
        "--bare",
        "--initial-branch=main",
        str(dot_git_path),
    )
    ensure_cmd_succeeds(git_init_bare_cmd)
    return dot_git_path


@pytest.fixture
def upstream_bare_repo(tmp_path: Path) -> Path:
    """A bare repository simulating the upstream project on GitHub."""
    return _make_bare_repo(tmp_path / "upstream.git")


@pytest.fixture
def fork_bare_repo(tmp_path: Path) -> Path:
    """A bare repository simulating the contributor's fork on GitHub."""
    return _make_bare_repo(tmp_path / "fork.git")


def _get_current_commit_hash() -> str:
    git_get_current_commit_cmd = "git", "rev-parse", "HEAD"
    return ensure_cmd_succeeds(git_get_current_commit_cmd).decode("utf-8").strip()


@pytest.fixture
def contrib_repo_work_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    upstream_bare_repo: Path,
    fork_bare_repo: Path,
    upstream_remote_name: str,
    pr_remote_name: str,
) -> Path:
    """An initialized Git checkout simulating a contributor's clone.

    The upstream remote points at the bare upstream repository and the
    PR remote at the bare fork repository, mirroring the documented
    ``cherry_picker --pr-remote=fork --upstream-remote=origin ...``
    invocation.
    """
    contrib_git_checkout = tmp_path / "simulated-contrib-workdir"

    git_init_cmd = (
        "git",
        "init",
        "--initial-branch=main",
        str(contrib_git_checkout),
    )
    ensure_cmd_succeeds(git_init_cmd)

    monkeypatch.chdir(contrib_git_checkout)

    git_commit_empty_cmd = "git", "commit", "-m", "✨ Initial commit", "--allow-empty"
    ensure_cmd_succeeds(git_commit_empty_cmd)

    git_remote_add_cmds = (
        ("git", "remote", "add", upstream_remote_name, str(upstream_bare_repo)),
        ("git", "remote", "add", pr_remote_name, str(fork_bare_repo)),
    )
    for git_remote_add_cmd in git_remote_add_cmds:
        ensure_cmd_succeeds(git_remote_add_cmd)

    return contrib_git_checkout


@pytest.fixture
def cherry_picker_config_path(contrib_repo_work_dir: Path) -> str:
    """Commit a minimal cherry-picker config and return its reference.

    The returned value is a Git revision specifier followed by a colon
    and an in-tree path, as accepted by the ``--config-path`` option.
    """
    initial_commit_hash = _get_current_commit_hash()
    cfg_path = Path(".cherry_picker.toml")  # in-tree, must be committed
    cfg_path.write_text(f'check_sha = "{initial_commit_hash}"', encoding="utf-8")

    git_add_config_cmd = "git", "add", str(cfg_path)
    ensure_cmd_succeeds(git_add_config_cmd)

    git_commit_config_cmd = (
        "git",
        "commit",
        "-m",
        "Add initial configuration file",
        str(cfg_path),
    )
    ensure_cmd_succeeds(git_commit_config_cmd)

    return f"{_get_current_commit_hash()}:{cfg_path}"


@pytest.fixture
def sha_hash_to_backport(
    contrib_repo_work_dir: Path,
    cherry_picker_config_path: str,
    upstream_remote_name: str,
) -> str:
    """Make a published target branch and a conflicting commit to pick.

    The target maintenance branch is pushed to the bare upstream
    repository so that cherry-picker can fetch it back, while an
    orphan branch gets a commit conflicting with the target branch.

    Returns the hash of the commit to backport.
    """
    conflicting_file_path = contrib_repo_work_dir / "conflicting-file"
    conflicting_file_path.write_text("original line", encoding="utf-8")

    git_switch_to_new_pick_branch_cmd = (  # git switch -c
        "git",
        "checkout",
        "-b",
        TARGET_BACKPORT_BRANCH,
    )
    ensure_cmd_succeeds(git_switch_to_new_pick_branch_cmd)

    git_add_conflicting_file_cmd = "git", "add", str(conflicting_file_path)
    ensure_cmd_succeeds(git_add_conflicting_file_cmd)

    git_commit_conflicting_file_cmd = (
        "git",
        "commit",
        "-m",
        "Add original conflicting file",
        str(conflicting_file_path),
    )
    ensure_cmd_succeeds(git_commit_conflicting_file_cmd)

    git_publish_branches_cmd = (
        "git",
        "push",
        upstream_remote_name,
        "main",
        TARGET_BACKPORT_BRANCH,
    )
    ensure_cmd_succeeds(git_publish_branches_cmd)

    git_make_orphan_branch_cmd = (  # git switch --orphan
        "git",
        "checkout",
        "--orphan",
        "another-branch",
    )
    ensure_cmd_succeeds(git_make_orphan_branch_cmd)

    conflicting_file_path.write_text("conflicting line", encoding="utf-8")

    ensure_cmd_succeeds(git_add_conflicting_file_cmd)
    git_commit_changed_conflicting_file_cmd = (
        "git",
        "commit",
        "-m",
        "Add changed conflicting file",
        str(conflicting_file_path),
    )
    ensure_cmd_succeeds(git_commit_changed_conflicting_file_cmd)

    return _get_current_commit_hash()


def _resolve_cherry_pick_conflict(
    contrib_repo_work_dir: Path,
    *,
    then_continue: bool = False,
) -> None:
    """Resolve the conflict and stage the result, as a human would.

    If ``then_continue`` is set, also run ``git cherry-pick --continue`` to
    finish the pick as a commit -- the second documented resolution style.
    """
    conflicting_file_path = contrib_repo_work_dir / "conflicting-file"
    conflicting_file_path.write_text("conflicting line", encoding="utf-8")
    git_add_conflicting_file_cmd = "git", "add", str(conflicting_file_path)
    ensure_cmd_succeeds(git_add_conflicting_file_cmd)

    if then_continue:
        git_cherry_pick_continue_cmd = "git", "cherry-pick", "--continue"
        ensure_cmd_succeeds(git_cherry_pick_continue_cmd)


def _assert_pushed_backport_commit_message(
    fork_bare_repo: Path,
    backport_branch: str,
    *,
    expected_pr_title: str,
    expected_pick_trailer: str,
) -> None:
    """Assert the branch pushed to the fork carries the amended message.

    Shared because both the auto-PR and ``--no-auto-pr`` resume paths run
    the same commit-message composition -- only whether a GitHub PR gets
    opened for it differs.
    """
    git_fork_branch_message_cmd = (
        "git",
        "--git-dir",
        str(fork_bare_repo),
        "log",
        "-1",
        "--format=%B",
        backport_branch,
    )
    raw_pushed_commit_message = ensure_cmd_succeeds(git_fork_branch_message_cmd)
    pushed_commit_message = raw_pushed_commit_message.decode("utf-8")
    assert pushed_commit_message.startswith(f"{expected_pr_title}\n")
    assert expected_pick_trailer in pushed_commit_message
    assert EXPECTED_COAUTHOR_TRAILER in pushed_commit_message


def _assert_workflow_cleaned_up(backport_branch: str) -> None:
    """Assert post-``--continue`` branch and Git-config state cleanup ran.

    Shared because a successful ``continue_cherry_pick()`` always checks
    the previous branch back out, deletes the local backport branch, and
    clears Git-config-backed workflow state, regardless of ``--auto-pr``.
    """
    assert cherry_picker.get_current_branch() == "another-branch"
    git_list_backport_branch_cmd = "git", "branch", "--list", backport_branch
    assert not ensure_cmd_succeeds(git_list_backport_branch_cmd).strip()

    assert cherry_picker.get_state() is cherry_picker.WORKFLOW_STATES.UNSET
    assert cherry_picker.load_val_from_git_cfg("previous_branch") is None
    assert cherry_picker.load_val_from_git_cfg("config_path") is None


@_parametrize_resolution_style
def test_resume_post_conflict_resolution_preserves_pr_title(
    browser_mock: MockType,
    cherry_picker_config_path: str,
    contrib_repo_work_dir: Path,
    fork_bare_repo: Path,
    gh_create_pr_http_mock: MockType,
    invoke_cherry_picker: Callable[..., CliResult],
    pr_remote_name: str,
    resolve_with_cherry_pick_continue: bool,
    sha_hash_to_backport: str,
    upstream_remote_name: str,
) -> None:
    """Auto-created PR keeps title and body after a conflict pause.

    Both documented conflict resolution styles must end up with the
    same amended commit message reaching the GitHub PR creation call:
    staging the resolution with ``git add`` alone, and additionally
    concluding the pick with ``git cherry-pick --continue``.
    """
    backport_branch = f"backport-{sha_hash_to_backport[:7]}-{TARGET_BACKPORT_BRANCH}"

    backporting_start_result = invoke_cherry_picker(
        f"--pr-remote={pr_remote_name}",
        f"--upstream-remote={upstream_remote_name}",
        f"--config-path={cherry_picker_config_path}",
        sha_hash_to_backport,
        TARGET_BACKPORT_BRANCH,
    )

    expected_conflict_message = (
        f"Failed to cherry-pick {sha_hash_to_backport} "
        f"into {TARGET_BACKPORT_BRANCH} ☹"
    )
    assert backporting_start_result.exit_code == -1
    assert expected_conflict_message in backporting_start_result.output

    paused_state = cherry_picker.get_state()
    assert paused_state is cherry_picker.WORKFLOW_STATES.BACKPORT_PAUSED
    stored_previous_branch = cherry_picker.load_val_from_git_cfg("previous_branch")
    assert stored_previous_branch == "another-branch"

    gh_create_pr_http_mock.assert_not_called()
    browser_mock.assert_not_called()

    _resolve_cherry_pick_conflict(
        contrib_repo_work_dir,
        then_continue=resolve_with_cherry_pick_continue,
    )

    backporting_completion_result = invoke_cherry_picker(
        f"--pr-remote={pr_remote_name}",
        f"--upstream-remote={upstream_remote_name}",
        f"--config-path={cherry_picker_config_path}",
        "--continue",
    )
    assert backporting_completion_result.exit_code == 0
    assert "Backport PR created at" in backporting_completion_result.output

    expected_pr_title = f"[{TARGET_BACKPORT_BRANCH}] Add changed conflicting file"
    expected_pick_trailer = f"(cherry picked from commit {sha_hash_to_backport})"

    gh_create_pr_http_mock.assert_called_once()
    expected_pr_api_url = "https://api.github.com/repos/python/cpython/pulls"
    assert gh_create_pr_http_mock.call_args.args == (expected_pr_api_url,)

    gh_pr_request_payload = gh_create_pr_http_mock.call_args.kwargs["json"]
    assert gh_pr_request_payload["title"] == expected_pr_title
    gh_pr_request_body = gh_pr_request_payload["body"]
    assert gh_pr_request_body.startswith(expected_pick_trailer)
    assert EXPECTED_COAUTHOR_TRAILER in gh_pr_request_body
    assert gh_pr_request_payload["base"] == TARGET_BACKPORT_BRANCH
    assert gh_pr_request_payload["head"].endswith(f":{backport_branch}")
    assert gh_pr_request_payload["draft"] is False
    assert gh_pr_request_payload["maintainer_can_modify"] is True

    gh_api_request_headers = gh_create_pr_http_mock.call_args.kwargs["headers"]
    assert gh_api_request_headers["authorization"] == "token sentinel-token-stub"

    _assert_pushed_backport_commit_message(
        fork_bare_repo,
        backport_branch,
        expected_pr_title=expected_pr_title,
        expected_pick_trailer=expected_pick_trailer,
    )
    _assert_workflow_cleaned_up(backport_branch)

    browser_mock.assert_not_called()


@_parametrize_resolution_style
def test_resume_post_conflict_resolution_no_auto_pr(
    browser_mock: MockType,
    cherry_picker_config_path: str,
    contrib_repo_work_dir: Path,
    fork_bare_repo: Path,
    gh_create_pr_http_mock: MockType,
    invoke_cherry_picker: Callable[..., CliResult],
    pr_remote_name: str,
    resolve_with_cherry_pick_continue: bool,
    sha_hash_to_backport: str,
    upstream_remote_name: str,
) -> None:
    """Resuming with ``--no-auto-pr`` pushes but never calls GitHub.

    Mirrors ``test_resume_post_conflict_resolution_preserves_pr_title``
    (same two documented conflict resolution styles, same push/cleanup
    assertions), but with ``--no-auto-pr``: the pushed branch must still
    carry the amended commit message and its trailers, while neither the
    GitHub API nor a web browser may ever be reached.
    """
    backport_branch = f"backport-{sha_hash_to_backport[:7]}-{TARGET_BACKPORT_BRANCH}"

    backporting_start_result = invoke_cherry_picker(
        f"--pr-remote={pr_remote_name}",
        f"--upstream-remote={upstream_remote_name}",
        f"--config-path={cherry_picker_config_path}",
        "--no-auto-pr",
        sha_hash_to_backport,
        TARGET_BACKPORT_BRANCH,
    )
    assert backporting_start_result.exit_code == -1
    paused_state = cherry_picker.get_state()
    assert paused_state is cherry_picker.WORKFLOW_STATES.BACKPORT_PAUSED

    _resolve_cherry_pick_conflict(
        contrib_repo_work_dir,
        then_continue=resolve_with_cherry_pick_continue,
    )

    backporting_completion_result = invoke_cherry_picker(
        f"--pr-remote={pr_remote_name}",
        f"--upstream-remote={upstream_remote_name}",
        f"--config-path={cherry_picker_config_path}",
        "--no-auto-pr",
        "--continue",
    )
    assert backporting_completion_result.exit_code == 0

    expected_pr_title = f"[{TARGET_BACKPORT_BRANCH}] Add changed conflicting file"
    expected_pick_trailer = f"(cherry picked from commit {sha_hash_to_backport})"
    assert expected_pr_title in backporting_completion_result.output

    _assert_pushed_backport_commit_message(
        fork_bare_repo,
        backport_branch,
        expected_pr_title=expected_pr_title,
        expected_pick_trailer=expected_pick_trailer,
    )
    _assert_workflow_cleaned_up(backport_branch)

    gh_create_pr_http_mock.assert_not_called()
    browser_mock.assert_not_called()
