当前复杂类别：**{category}**

视频信息：时长 {duration}秒，路径 {video_path}

简单类别先验：
{merged_simple_results}

本类别重点目标：
{focus_targets}

本类别忽略项：
{ignore_targets}

操作步骤：
1. 全局浏览完整视频，只关注【重点目标】（忽略「忽略项」中的对象）；
2. 通过 submit_time_plan 提交（skip/time_slices 规则详见工具参数说明）；
3. suspected_evidence：只描述**可见客观事实**（目标种类、出现位置、朝向/移动方向），**禁止输出行为结论或意图推断**。
   - **运动归因**：画面中目标变大/变近不等于目标在移动。必须区分「目标自身主动移动」与「自车前行导致目标在画面中变大」。如果目标相对地面静止，即使画面中越来越近，suspected_evidence 中也应如实描述为「静止车辆」而非「向本车驶来」。
4. relevant_context：从上方简单类别先验中摘取环境信息，禁止捏造。
