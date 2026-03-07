from __future__ import annotations

from pathlib import Path

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError



def pull_current_branch(repo_path: Path) -> str:
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError as exc:
        raise RuntimeError("La carpeta no es un repositorio Git válido.") from exc

    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise RuntimeError("No se encontró .git en el directorio del proyecto.")

    if repo.is_dirty(untracked_files=True):
        raise RuntimeError(
            "Hay cambios locales sin confirmar. Hacé commit/stash antes de actualizar."
        )

    branch = repo.active_branch.name
    try:
        output = repo.git.pull("origin", branch)
    except GitCommandError as exc:
        raise RuntimeError(f"Error durante git pull: {exc}") from exc

    return f"Actualización completada ({branch}): {output}"
