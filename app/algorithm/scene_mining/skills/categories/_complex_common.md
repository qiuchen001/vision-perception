## 约束

- step0、step0b 和 step3 必须非空
- 所有 step 字段必须非空，体现视觉证据与时序
- pred 优先从候选子类中选择；出现未列入的 corner case 异常时，用简短中文描述该 case（≤10字），同时在 events 中保留具体描述，绝不可仅返回 pred: ["corner case"] 而无具体内容
- 同一时间窗口内可能同时存在多种异常事件，pred 允许多选，每种异常都必须在 events 中给出独立的 start_time / end_time
- events 中每个 type 必须来自 pred
- start_time <= end_time，且不小于 0
- 只关注被类别应该关注的对象目标

## JSON Schema

{
  "step0_environment_context": "先验知识分析（道路类型、天气、光照等）",
  "step0b_motion_reference": "① 自车[行驶中/静止]；② 本类别关键目标的运动状态与绝对方向；③ 区分「目标主动靠近本车」vs「自车前行接近目标」",
  "step1_object_detection": "识别相关目标：[车道位置]+[近/中/远距]+外观及初始状态",
  "step2_motion_analysis": "基于step0b参考系，分析各目标绝对运动轨迹与时序变化",
  "step3_conflict_check": "确认事件是否实际发生，描述已发生的交互事实",
  "pred": "子类中文名数组（优先选候选子类；未列入的异常用≤10字简短描述）",
  "events": [{"type": "子类中文名", "start_time": 秒数, "end_time": 秒数}]
}