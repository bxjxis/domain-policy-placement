# 已存八组合结果的恢复汇总

`summarize.py` 只读已有数据，无需 torch、GPU 或重新生成。兼容旧用法：

```powershell
python summarize.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct --data-dir "实际结果目录" --json-out summary.json --csv-out summary.csv
```

不传 `--data-dir` 时使用用户目录下的 `orr_spike`。也可通过多个 `--combo` 指定结果文件，通过 `--input-results` 指定旧 `health_orsc_results.json`。默认只打印，不写文件；不会允许输出覆盖源文件。

## 恢复方法与校验

- 按原始 prompt 精确匹配，不依赖记录顺序；同 prompt 的重复配置或缺失配置报错，不悄悄去重。
- 从 `P1R0O0` 的实际 stop 恢复输入阻断；旧结果的 `input_verdict` 必须与它相符，才能恢复 `input_cats`。共享类别的前提是输入 guard、模板、策略和解码设置相同，不只是模型无关。
- 从 `P0R0O0` / `P0R1O0` 恢复两套完整的模型自拒，从 `P0R0O1` / `P0R1O1` 恢复输出判定与类别。
- 用恢复原子重建八组，并核对存盘 stop、最终标签、可观察的自拒和输出标志。仅忽略已识别的旧 bug：通过输入检查后在输出 stop 处错误保存 `input_blocked=True`。原始文件不修改。
- 比较多个模型时要求 prompt 集合完全相同。未知类别产生 `null` / `NA` 和未知数量，不默认为非 S6。

## 字段含义

| 字段 | 含义 |
|---|---|
| `no_answer` | 管线最终关键词拒绝或 guard 阻断数；不是正式 ORR |
| `atomic_self_refusal` | 全部请求在相应 R 下的模型自拒，不随 P/O 改变 |
| `observed_self_refusal` | 实际到达模型的请求中自拒数，包含后来被 O 阻断的 |
| `surviving_self_refusal` | 最终未被 guard 阻断、交付模型拒绝回答的数量 |
| `new_from_input` | P 阻断且离线模型本来不拒绝的请求数；不代表本来会合规回答 |
| `new_from_output` | 到达 O、被 O 阻断且模型此前未拒绝的请求数 |
| `guard_blocks` | 首个实际 guard stop 的总数 |
| `stopped_non_s6_only` | 原管线实际 stop 中，类别不只有 S6 的数量 |
| `no_answer_excl_s6_only` | 绕过所有 S6-only guard 判定后，重新走完整顺序的无回答数 |
| `guard_blocks_excl_s6_only` | 同一反事实管线的 guard stop 数 |
| `unknown_stop_categories` | 原管线 stop 的类别未知数量 |
| `unknown_s6_replay` | 反事实重建因关键类别未知而无法判定的请求数 |

每组满足：`no_answer = atomic_self_refusal + new_from_input + new_from_output`。

“当前 stop 剔除 S6”与“绕过 S6 后重走管线”不同。例如输入因 S6 拦住，但输出因 S2 也会拦住，绕过输入 S6 后仍然无回答。

JSON 另存两类分析：R=0/1 的逐条拒绝转移计数，以及每个 R 的输入/输出、输入/模型自拒、输出/模型自拒 shadow 重叠。它们只能说明机制与关键词拒绝，无法代替原文人工判定、策略 ORR、ASR 或正式配对显著性分析。

本地验证：`python -m unittest test_summarize.py`。测试使用合成数据；真实结果仍需在持有结果文件的机器上运行。
