from pathlib import Path
from subprocess import check_output as ensure_cmd_succeeds
# import typing as _t

import pytest

from click.testing import CliRunner

from pytest_mock import MockFixture

from . import cherry_picker
from . import test_cherry_picker  # init fixtures


@pytest.fixture
def upstream_bare_repo(tmp_path: Path) -> Path:
    dot_git_path = tmp_path / 'upstream.git'
    dot_git_path.mkdir()

    git_init_cmd = (
        "git",
        "init",
        "--bare",
        "--initial-branch=main",
        str(dot_git_path),
    )
    ensure_cmd_succeeds(git_init_cmd)

    return dot_git_path


@pytest.fixture
def upstream_remote_name() -> str:
    return "origin"


@pytest.fixture
def contrib_repo_work_dir(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        upstream_bare_repo: Path,
        upstream_remote_name: str,
) -> Path:
    contrib_git_checkout = tmp_path / 'simulated-contrib-workdir'

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

    git_remote_add_cmd = (
        "git",
        "remote",
        "add",
        upstream_remote_name,
        str(contrib_git_checkout),
    )
    ensure_cmd_succeeds(git_remote_add_cmd)

    return contrib_git_checkout


@pytest.fixture
def cherry_picker_config_path(
        contrib_repo_work_dir: Path,
) -> str:
    git_get_current_commit_cmd = 'git', 'rev-parse', 'HEAD'
    initial_commit_hash = ensure_cmd_succeeds(git_get_current_commit_cmd).decode('utf-8').strip()  # extract the commit hash <- git rev-parse HEAD
    cfg_path = Path('.cherry_picker.toml')  # in-tree, must be committed
    cfg_path.write_text(f'check_sha = "{initial_commit_hash}"', encoding="utf-8")

    git_add_config_cmd = "git", "add", str(cfg_path)
    ensure_cmd_succeeds(git_add_config_cmd)

    git_commit_config_cmd = "git", "commit", "-m", "Add initial configuration file", str(cfg_path)
    ensure_cmd_succeeds(git_commit_config_cmd)

    config_commit_hash = ensure_cmd_succeeds(git_get_current_commit_cmd).decode('utf-8').strip()  # extract the commit hash <- git rev-parse HEAD

    return f'{config_commit_hash}:{cfg_path}'


def test_resume_post_conflict_resolution_preserves_pr_title(
        contrib_repo_work_dir: Path,
        cherry_picker_config_path: str,
        mocker: MockFixture,
        monkeypatch: pytest.MonkeyPatch,
        upstream_remote_name: str,
) -> None:
    target_backport_branch = "3.14"
    conflicting_file_path = contrib_repo_work_dir / 'conficting-file'
    conflicting_file_path.write_text('original line')

    git_switch_to_new_pi_branch_cmd = "git", "checkout", "-b", target_backport_branch  # git switch -c
    ensure_cmd_succeeds(git_switch_to_new_pi_branch_cmd)

    git_add_conficting_file_cmd = "git", "add", str(conflicting_file_path)
    ensure_cmd_succeeds(git_add_conficting_file_cmd)

    git_commit_conficting_file_cmd = "git", "commit", "-m", "Add original conflicting file", str(conflicting_file_path)
    ensure_cmd_succeeds(git_commit_conficting_file_cmd)

    git_make_orphan_branch_cmd = "git", "checkout", "--orphan", "another-branch"  # git switch --orphan
    ensure_cmd_succeeds(git_make_orphan_branch_cmd)

    conflicting_file_path.write_text('conflicting line')

    ensure_cmd_succeeds(git_add_conficting_file_cmd)
    git_commit_conficting_file_cmd = "git", "commit", "-m", "Add changed conflicting file", str(conflicting_file_path)
    ensure_cmd_succeeds(git_commit_conficting_file_cmd)
    git_get_current_commit_cmd = 'git', 'rev-parse', 'HEAD'
    sha_hash_to_backport = ensure_cmd_succeeds(git_get_current_commit_cmd).decode('utf-8').strip()  # extract the commit hash <- git rev-parse HEAD
    print(f'{sha_hash_to_backport=}')
    click_app_runner = CliRunner()
    # cherry_picker --pr-remote=fork --upstream-remote=origin --no-auto-pr 78b1370de96bab5eee82b022a9e26e633a040e0b 3.13
    backporting_start_result = click_app_runner.invoke(
        cherry_picker.cherry_pick_cli,
        (
            f'--pr-remote={upstream_remote_name}',
            f'--config-path={cherry_picker_config_path}',
            # '--no-auto-pr',
            '--auto-pr',
            sha_hash_to_backport,
            target_backport_branch,
        ),
    )
    assert isinstance(backporting_start_result.exception, SystemExit)
    assert f'Failed to cherry-pick {sha_hash_to_backport} into {target_backport_branch} ☹\n' in backporting_start_result.output

    conflicting_file_path.write_text('conflicting line')  # (1) conflict resolution
    ensure_cmd_succeeds(git_add_conficting_file_cmd)  # (2) conflict resolution

    # monkeypatch.setattr(cherry_picker, 'CREATE_PR_URL_TEMPLATE', 'http://localhost')
    monkeypatch.setenv('GH_AUTH', 'sentinel-token-stub')
    # mocker.patch.object(cherry_picker.requests, 'post', mocker.Mock())

    # mocker.patch.object(cherry_picker.webbrowser, 'open_new_tab', mocker.Mock())
    create_gh_pr_mock = mocker.patch.object(cherry_picker.CherryPicker, 'create_gh_pr', mocker.Mock())
    # cherry_picker --pr-remote=fork --upstream-remote=origin --no-auto-pr --continue
    backporting_completion_result = click_app_runner.invoke(
        cherry_picker.cherry_pick_cli,
        (
            f'--pr-remote={upstream_remote_name}',
            f'--config-path={cherry_picker_config_path}',
            # '--no-auto-pr',
            '--auto-pr',
            '--continue',
        ),
    )
    assert backporting_completion_result.exit_code == 0
    assert 'Backport PR:' in backporting_completion_result.output
    assert create_gh_pr_mock.call_args.kwargs['commit_message'].startswith(f'[{target_backport_branch}] ')
    # breakpoint()

    # given a bare repo (via the fixture) and a checkout simulation (via another fixture)
    # simulate a commit to "backport" + a conflicting commit
    # given a commit that conflicts with the last branch
    # cherry-picker --no-auto-pr would pause
    # the state should be asserted
    # fix the "conflict"
    # git add
    # mock the API call<<
    # cherry-picker --no-auto-pr --continue
    # the state should be asserted again
    # check the API call has correct title
    # ...
