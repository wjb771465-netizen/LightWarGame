# source 这个文件以获得 git_snapshot 函数。
# 用法：git_snapshot <tag>
# 将当前 HEAD 代码树提交到 exp 分支并打 tag exp/<tag>。
# 失败时打印警告但不中断训练。

git_snapshot() {
    local tag_name="$1"
    (
        set +e
        git rev-parse --git-dir > /dev/null 2>&1 || { echo "[snapshot] 不在 git 仓库，跳过" >&2; exit 0; }

        local dirty_note=""
        if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
            echo "[snapshot] 工作区有未提交改动，snapshot 仅捕获 HEAD" >&2
            dirty_note=" [dirty]"
        fi

        local head_tree
        head_tree=$(git rev-parse "HEAD^{tree}") || { echo "[snapshot] 无法读取 HEAD tree" >&2; exit 0; }

        local commit
        if git rev-parse --verify exp > /dev/null 2>&1; then
            commit=$(git commit-tree "$head_tree" -p "$(git rev-parse exp)" -m "exp: ${tag_name}${dirty_note}")
        else
            commit=$(git commit-tree "$head_tree" -m "exp: ${tag_name}${dirty_note}")
        fi
        [[ -n "$commit" ]] || { echo "[snapshot] commit-tree 失败" >&2; exit 0; }

        git branch -f exp "$commit"
        git tag "exp/${tag_name}" "$commit"
        echo "[snapshot] exp/${tag_name} -> ${commit:0:8}"
    ) || echo "[snapshot] 快照失败（不影响训练）" >&2
}
