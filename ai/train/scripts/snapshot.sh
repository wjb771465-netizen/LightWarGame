# source 这个文件以获得 git_snapshot 函数。
# 用法：git_snapshot <tag>
# 将当前工作区完整状态（含未提交改动）提交到 exp 分支并打 tag exp/<tag>。
# 使用临时 index 文件，完全不碰主分支的暂存区和工作区。
# 失败时打印警告但不中断训练。

git_snapshot() {
    local tag_name="$1"
    (
        set +e
        git rev-parse --git-dir > /dev/null 2>&1 || { echo "[snapshot] 不在 git 仓库，跳过" >&2; exit 0; }

        local dirty=""
        if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
            dirty=" [dirty]"
        fi

        # 用临时 index 暂存工作区全部改动，不影响真实 index
        local tmp_index
        tmp_index=$(mktemp) || { echo "[snapshot] 无法创建临时 index" >&2; exit 0; }
        trap 'rm -f "$tmp_index"' RETURN

        # 将真实 index 拷贝到临时 index，然后 add -A 全部改动
        if ! GIT_INDEX_FILE="$tmp_index" git read-tree HEAD 2>/dev/null; then
            # read-tree 失败（可能是空仓库），尝试从空 index 开始
            : > "$tmp_index"
        fi
        GIT_INDEX_FILE="$tmp_index" git add -A > /dev/null 2>&1

        local head_tree
        head_tree=$(GIT_INDEX_FILE="$tmp_index" git write-tree) || {
            echo "[snapshot] write-tree 失败" >&2
            exit 0
        }

        local commit
        if git rev-parse --verify exp > /dev/null 2>&1; then
            commit=$(git commit-tree "$head_tree" -p "$(git rev-parse exp)" -m "exp: ${tag_name}${dirty}")
        else
            commit=$(git commit-tree "$head_tree" -m "exp: ${tag_name}${dirty}")
        fi
        [[ -n "$commit" ]] || { echo "[snapshot] commit-tree 失败" >&2; exit 0; }

        git branch -f exp "$commit"
        git tag "exp/${tag_name}" "$commit"
        echo "[snapshot] exp/${tag_name} -> ${commit:0:8}"
        git push -f origin exp 2>/dev/null && git push origin "exp/${tag_name}" 2>/dev/null \
            && echo "[snapshot] 已推送到 origin" >&2 \
            || echo "[snapshot] push 失败（不影响训练）" >&2
    ) || echo "[snapshot] 快照失败（不影响训练）" >&2
}
