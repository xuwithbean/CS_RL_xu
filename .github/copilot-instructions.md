# Copilot 指令 — CS_RL_xu

下面是帮助 AI 编码助手（例如 GitHub Copilot / 本地 coding agent）快速在此代码库中高效工作的简洁指引。保持短小、可执行，并引用仓库中已有的示例文件。

1. 项目概览（大局观）:
   - **目标**: 本仓库是一个毕业设计项目，目标是使用强化学习（RL）来玩 CS 游戏（参见 `README.md`）。
   - **运行环境提示**: Windows + WSL2 的混合环境。
2. 关键文件与示例模式（引用与约定）:
   - **`get_screenshot.py`**: 负责从操作系统获取游戏画面。文件开头有说明：截图在 Windows 上完成并通过 WSL2 读取。示例函数名：`get_screenshot(save_path)`。
3. 项目约定与风格指引（仅限可被仓库内容验证的约定）:
   - **注释与文档**: 保持现有注释风格，尤其是中文注释。新函数应在顶部添加简要说明，解释输入输出和功能。
   - **函数命名**: 多数模块采用 `get_` 前缀（例如 `get_screenshot`, `get_policy`, `get_action`），遵循该命名风格以保持一致性。
   - **跨平台注意**: 因注释说明使用 Windows 截图并在 WSL2 中读取，避免在代码中硬编码 Windows 专有路径；若需绝对路径，使用环境变量或配置文件。
   - **轻量修改优先**: 仓库目前是原型/骨架，代写或修改时优先做最小改动以实现功能，不要进行大规模重构或引入复杂框架。

4. 运行、调试与常用命令:
   - 激活 Conda 环境:
     - `conda activate condacommon`
   - 运行训练/测试脚本:
     - `/home/xu/anaconda3/envs/condacommon/bin/python /home/xu/code/CS_RL_xu/train.py`
     - `/home/xu/anaconda3/envs/condacommon/bin/python /home/xu/code/CS_RL_xu/test.py`

