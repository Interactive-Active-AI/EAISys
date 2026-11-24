const SNAP = window.__EAISYS_SNAPSHOT__ || {};

const STATE = {
  lang: "en",
  taskFilter: "all",
  selectedRunId: null,
  openTurn: 1,
  skillTab: "motion",
  selectedSkill: "MV_FWD",
  robotFilter: "all",
  cliTab: "verify",
  priorQuery: "",
  simMode: "urdf6",
  simAuto: true,
};

/* ── Plain-Language Bilingual Copy ────────────────────────────────────── */
const COPY = {
  en: {
    nav_demos: "3D Demos",
    nav_skills: "Skill Library",
    nav_robots: "15 Robots",
    nav_start: "Try It",
    menu_btn: "Menu",
    hero_kicker: "Cross-Embodiment Robot Agent",
    hero_title: "One brain for any robot body.",
    hero_lead:
      "Hand it a robot's URDF manual and it starts working on the new arm right away. Hand it nothing at all — it wiggles its own joints for a few seconds, figures out how its body moves, and still finishes the job.",
    p1_title: "Have a URDF? Zero-shot ready.",
    p1_desc: "Reads the robot's file, writes & verifies its own controller, then operates single or dual arms straight away.",
    p2_title: "No URDF? Self-learns in seconds.",
    p2_desc: "Wiggles fingers to spot its hand, nudges each joint, estimates its motion map, and adapts live.",
    cta_demos: "Watch 3D Demos",
    cta_skills: "Explore Skill Library",
    sim_urdf6: "6-DoF · URDF",
    sim_urdf7: "7-DoF · URDF",
    sim_blind: "No URDF · Probe",
    cmd_back: "Back",
    cmd_up: "Up",
    cmd_down: "Down",
    cmd_fwd: "Forward",
    cmd_grasp: "Grasp",
    cmd_auto: "Auto",
    stat_videos_lbl: "3D tasks solved, zero cheating",
    stat_robots_lbl: "Different robots verified",
    stat_skills_lbl: "Universal skill commands",
    stat_demos_lbl: "Human demonstrations needed",
    sec1_kicker: "01 · WATCH IT WORK",
    sec1_title: "Real 3D tasks. Eyes and joints only.",
    sec1_lead:
      "No hidden object coordinates, no ruler overlays, no simulator cheats. Pick any task below to watch the live 3D video and step through what the AI saw and decided.",
    sec2_kicker: "02 · UNIVERSAL SKILL LIBRARY",
    sec2_title: "One shared skill library for every body.",
    sec2_lead:
      "Instead of teaching every new robot from zero, EAISys separates high-level skills from hardware: 25 universal motion primitives, a shared 244-line kinematic engine, reusable manipulation habits, and a built-in self-discovery skill when no URDF exists.",
    sec3_kicker: "03 · SUPPORTED ROBOTS",
    sec3_title: "15 different robots, one clean interface.",
    sec3_lead:
      "Single arms, dual arms, mobile platforms, 3-finger hands, and swapped tools. Click any robot to compare its raw URDF file alongside the Python controller EAISys generated for it.",
    sec4_kicker: "04 · TRY IT YOURSELF",
    sec4_title: "Up and running in minutes.",
    sec4_lead: "Works with OpenAI, OpenRouter, local vLLM, Anthropic Claude, or Codex CLI.",
    compare_toggle: "How does EAISys compare with end-to-end VLAs and other robot agents?",
    footer_note: "Open-Source Cross-Embodiment Robot Harness · Apache-2.0",
    footer_docs: "Docs",
  },
  zh: {
    nav_demos: "3D 实拍",
    nav_skills: "通用技能库",
    nav_robots: "15 款机器人",
    nav_start: "快速上手",
    menu_btn: "菜单",
    hero_kicker: "跨身体通用具身智能体",
    hero_title: "一套大脑，开箱控制任何机器人。",
    hero_lead:
      "给它一份机器人的 URDF 说明书，新机械臂直接就能零样本干活；什么都不给，让它自己试探关节几秒钟，摸清身体规律后照样能完成 3D 任务。",
    p1_title: "有 URDF？读完即用，零样本上手",
    p1_desc: "自动读懂机身说明书、写出并通过物理验算，单臂、换工具乃至异构双臂直接干活。",
    p2_title: "没 URDF？自己动几秒也能学会",
    p2_desc: "先抖指找到自己手在哪，再轻推每个关节估出运动规律，边做任务边在线校准。",
    cta_demos: "看 3D 实拍演示",
    cta_skills: "探索通用技能库",
    sim_urdf6: "6 轴 · 有 URDF",
    sim_urdf7: "7 轴 · 有 URDF",
    sim_blind: "无 URDF · 自主试探",
    cmd_back: "后退",
    cmd_up: "抬高",
    cmd_down: "下降",
    cmd_fwd: "前伸",
    cmd_grasp: "抓合",
    cmd_auto: "自动演示",
    stat_videos_lbl: "真实 3D 任务全通关（零作弊）",
    stat_robots_lbl: "款不同形态机器人通过验算",
    stat_skills_lbl: "个跨机身通用动作技能",
    stat_demos_lbl: "条人工示教（零样本 / 自试探）",
    sec1_kicker: "01 · 实拍演示",
    sec1_title: "真实 3D 物理任务：只看画面，不偷看坐标。",
    sec1_lead:
      "不给物体坐标、不画辅助尺、不作弊。点选下方任意任务，直接看完整 3D 视频，并一步步查看大模型当时看到了什么、怎么想的。",
    sec2_kicker: "02 · 通用技能库",
    sec2_title: "所有机器人共用一套通用技能库。",
    sec2_lead:
      "不必每换一台新机器人就从头训练。EAISys 把“怎么做事”和“身体长啥样”彻底解耦：25 个跨机身基础动作、一套 244 行通用运动引擎、可迁移操作经验，以及没说明书时的自主试探技能。",
    sec3_kicker: "03 · 已支持机器人",
    sec3_title: "15 款不同机器人，一套统一接口接管。",
    sec3_lead:
      "涵盖单臂、双臂、移动底盘、三指手与换装工具。点击任意卡片，左右对照查看原始 URDF 与 AI 自动生成的 Python 控制代码。",
    sec4_kicker: "04 · 快速开始",
    sec4_title: "三行命令，几分钟跑通。",
    sec4_lead: "不挑模型：直接支持 OpenAI、OpenRouter、本地 vLLM、Anthropic 与 Codex CLI。",
    compare_toggle: "想了解细节？看看它和传统 VLA / 其他开源具身框架的区别",
    footer_note: "开源跨身体具身智能框架 · Apache-2.0",
    footer_docs: "技术文档",
  },
};

/* ── Plain-Language 25 Universal Motion Skills Metadata ───────────────── */
const SKILL_GROUPS = [
  {
    key: "translate",
    titleEn: "Move Hand in 3D Space",
    titleZh: "空间 6 向平移",
    noteEn: "World-aligned hand motion",
    noteZh: "按当前步长档位平滑移动",
    items: [
      { id: "MV_FWD", mode: "move", vec: [1, 0, 0], en: "Reach forward (+X) away from the robot base.", zh: "沿正前方（+X）向前伸手。" },
      { id: "MV_BACK", mode: "move", vec: [-1, 0, 0], en: "Pull back (-X) toward the robot base.", zh: "沿后方（-X）向机身回撤。" },
      { id: "MV_LEFT", mode: "move", vec: [0, 1, 0], en: "Slide left (+Y) horizontally across the workspace.", zh: "沿左侧（+Y）水平横移。" },
      { id: "MV_RIGHT", mode: "move", vec: [0, -1, 0], en: "Slide right (-Y) horizontally across the workspace.", zh: "沿右侧（-Y）水平横移。" },
      { id: "MV_UP", mode: "move", vec: [0, 0, 1], en: "Lift straight upward (+Z) to clear walls or obstacles.", zh: "垂直向上（+Z）抬起以越过障碍或提物。" },
      { id: "MV_DOWN", mode: "move", vec: [0, 0, -1], en: "Lower gently downward (-Z) toward the target or surface.", zh: "垂直向下（-Z）轻柔下降接近目标或桌面。" },
    ],
  },
  {
    key: "rotate",
    titleEn: "Turn Wrist Orientation",
    titleZh: "手腕 3 轴旋转（每次 15°）",
    noteEn: "15° per step (SO(3))",
    noteZh: "精准对准门把手、锅柄与卡槽",
    items: [
      { id: "ROT_YAW_CW", mode: "rot", axis: "yaw", dir: 1, en: "Twist wrist clockwise around vertical axis to align gripper jaws with handles.", zh: "手腕顺时针偏航转动 15°，让夹爪跨过门把或锅柄。" },
      { id: "ROT_YAW_CCW", mode: "rot", axis: "yaw", dir: -1, en: "Twist wrist counter-clockwise around vertical axis.", zh: "手腕逆时针偏航转动 15°。" },
      { id: "ROT_PITCH_CW", mode: "rot", axis: "pitch", dir: 1, en: "Tilt wrist forward/downward by 15°.", zh: "手腕俯仰向前倾斜 15°。" },
      { id: "ROT_PITCH_CCW", mode: "rot", axis: "pitch", dir: -1, en: "Tilt wrist backward/upward by 15°.", zh: "手腕俯仰向后仰起 15°。" },
      { id: "ROT_ROLL_CW", mode: "rot", axis: "roll", dir: 1, en: "Roll wrist clockwise by 15°.", zh: "手腕顺时针滚转 15°。" },
      { id: "ROT_ROLL_CCW", mode: "rot", axis: "roll", dir: -1, en: "Roll wrist counter-clockwise by 15°.", zh: "手腕逆时针滚转 15°。" },
    ],
  },
  {
    key: "gear",
    titleEn: "Switch Precision Gear",
    titleZh: "3 档步长变速",
    noteEn: "4 cm / 2 cm / 1 cm",
    noteZh: "远距离快跑，近距离毫米级微调",
    items: [
      { id: "STEP_COARSE", mode: "gear", size: 4, en: "Fast 4 cm travel gear for crossing empty space quickly.", zh: "4 cm 粗步档：空旷区域快速穿梭搬运。" },
      { id: "STEP_FINE", mode: "gear", size: 2, en: "Balanced 2 cm approach gear when nearing an object.", zh: "2 cm 标准档：接近目标时稳健逼近。" },
      { id: "STEP_ULTRA_FINE", mode: "gear", size: 1, en: "1 cm precision gear for tight peg insertion and handle seating.", zh: "1 cm 精调档：精密套环插孔与把柄对准。" },
    ],
  },
  {
    key: "grip",
    titleEn: "Gripper & Task Flow",
    titleZh: "抓放与任务控制",
    noteEn: "Auto-adapts to jaw speed",
    noteZh: "自动等待夹爪编码器稳态",
    items: [
      { id: "GRASP", mode: "grip", closed: true, en: "Close fingers firmly and wait until jaw encoders settle.", zh: "稳固闭合夹爪，并等待硬件编码器稳定受力。" },
      { id: "RELEASE", mode: "grip", closed: false, en: "Open fingers smoothly to drop or release the object.", zh: "平滑张开夹爪放下物体或松开把柄。" },
      { id: "HOLD", mode: "hold", en: "Keep this arm steady in place (useful during dual-arm coordination).", zh: "保持当前臂静止稳住（双臂协同或观察时非常有用）。" },
      { id: "DONE", mode: "done", en: "Signal that the robot believes the goal is visibly complete.", zh: "声明任务在视觉上已全部完成。" },
    ],
  },
  {
    key: "base",
    titleEn: "Roll Mobile Base",
    titleZh: "移动底盘 SE(2) 行走",
    noteEn: "Enabled automatically for wheeled robots",
    noteZh: "检测到轮式底盘时自动启用",
    items: [
      { id: "BASE_FWD", mode: "base", vec: [1, 0, 0], en: "Drive mobile base forward by 15 cm.", zh: "底盘向前移动 15 cm。" },
      { id: "BASE_BACK", mode: "base", vec: [-1, 0, 0], en: "Drive mobile base backward by 15 cm.", zh: "底盘向后退 15 cm。" },
      { id: "BASE_LEFT", mode: "base", vec: [0, 1, 0], en: "Strafe mobile base left by 15 cm.", zh: "底盘向左平移 15 cm。" },
      { id: "BASE_RIGHT", mode: "base", vec: [0, -1, 0], en: "Strafe mobile base right by 15 cm.", zh: "底盘向右平移 15 cm。" },
      { id: "BASE_YAW_LEFT", mode: "base", vec: [0, 0, 1], en: "Turn mobile base left by 15°.", zh: "底盘原地左转 15°。" },
      { id: "BASE_YAW_RIGHT", mode: "base", vec: [0, 0, -1], en: "Turn mobile base right by 15°.", zh: "底盘原地右转 15°。" },
    ],
  },
];

const HABIT_CARDS = [
  {
    tag: "universal",
    tagZh: "通用习惯",
    titleEn: "Look Before You Touch",
    titleZh: "多视角确认再下压",
    bodyEn: "Never trust a single camera for depth. Cross-check front, side, and wrist views, shift gears from 4 cm to 1 cm near contact, and retract immediately if a joint stalls.",
    bodyZh: "绝不靠单一视角猜深度：俯视、侧视与腕部相机交叉核对，靠近目标自动降档到 1 cm 精调，遇碰撞阻滞立刻回撤。",
  },
  {
    tag: "bimanual",
    tagZh: "异构双臂",
    titleEn: "Two Different Arms, One Rhythm",
    titleZh: "异构双臂同步发力",
    bodyEn: "Move both arms symmetrically to opposite handles, sink slightly below parallel-linkage fingertips before grasping, and lift together so heavy objects stay level.",
    bodyZh: "两台不同机器人并行对齐左右把柄，考虑夹爪闭合上缩提前下探到位，再同步抬升保持重物平稳不滑脱。",
  },
  {
    tag: "insertion",
    tagZh: "精密装配",
    titleEn: "Precision Ring & Peg Alignment",
    titleZh: "精密套环与对孔装配",
    bodyEn: "Grab the outer tab so the inner hole stays visible, center the bore over the peg tip in both overhead and side views at 1 cm precision, then lower straight down.",
    bodyZh: "抓稳圆环外侧手柄、保持内孔清晰可见，在俯视与侧视下以 1 cm 精调对准圆柱销后垂直落座。",
  },
  {
    tag: "wiping",
    tagZh: "表面擦除",
    titleEn: "Feel Surface Contact When Wiping",
    titleZh: "非抓取擦除的触地判据",
    bodyEn: "Never sweep in mid-air: descend until vertical TCP progress stalls on the table and marks visibly vanish beneath the sponge, then sweep overlapping passes.",
    bodyZh: "避免悬空瞎擦：持续下探直到末端高度受桌面阻滞且污渍开始消失，再连续重叠擦净全部残留。",
  },
  {
    tag: "articulated",
    tagZh: "铰链开门",
    titleEn: "Doors & Hinged Handles",
    titleZh: "铰链门与拉手弧线",
    bodyEn: "Check clearance behind the bar before descending, rotate wrist yaw to clear wide jaws, and pull along the hinge arc in short verified steps.",
    bodyZh: "先确认把柄后方空隙，必要时旋转手腕避让，再顺着门轴弧线分段拉动并观察开门角度。",
  },
  {
    tag: "sorting",
    tagZh: "分拣跨墙",
    titleEn: "Clear Walls Before Carrying",
    titleZh: "翻越围挡与分格投放",
    bodyEn: "Lift until there is an unmistakable air gap underneath the held item in the front view before moving laterally over bin dividers into the matching slot.",
    bodyZh: "先抬高到主视角肉眼可见明显离地间隙，再跨过分拣箱隔板水平移至目标格上方平稳松手。",
  },
];

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function getDemoPlainMeta(d, z) {
  const fm = d.final_metrics || {};
  const tn = String(d.task_name || "");
  if (d.urdf_free || tn.includes("URDF_Free")) {
    return {
      title: z ? "无 URDF 盲探抓取（Panda）" : "Blind No-URDF Lift (Panda)",
      summary: z
        ? "完全没给 URDF 说明书。它先抖指找到手在哪，试探 7 个关节估出运动规律，随后 12 步稳稳抓起红方块。"
        : "Zero URDF provided. The arm wiggled its fingers to spot its hand, probed 7 joints to estimate motion, then lifted the red cube in 12 turns.",
      badge: "0-URDF · 7p",
      badgeCls: "purple",
      resultLbl: z ? "最终高度" : "Cube Lifted",
      resultVal: `z = ${((fm.cube_z_m || 0) * 100).toFixed(1)} cm`,
    };
  }
  if (tn === "TwoArmLift") {
    const cm = ((fm.pot_lift_clearance_m || 0) * 100).toFixed(1);
    return {
      title: z ? "异构双臂协同抬锅（Kinova3 + UR5e）" : "Two Different Arms Lift Pot",
      summary: z
        ? "左边 7 轴 Kinova3、右边 6 轴 UR5e，两份完全不同的裸 URDF 各自算好动作并同步抬起 1.2kg 红锅。"
        : "A 7-DoF Kinova3 and a 6-DoF UR5e — two completely different unseen URDFs — coordinate in real time to lift a heavy pot together.",
      badge: `+${cm}cm`,
      badgeCls: "rose",
      resultLbl: z ? "锅底离桌高度" : "Pot Clearance",
      resultVal: `+${cm} cm`,
    };
  }
  if (tn === "NutAssemblyRound") {
    const mm = ((fm.xy_alignment_err_m || 0) * 1000).toFixed(1);
    return {
      title: z ? "精密圆环套柱装配（UR5e）" : "Precision Ring-on-Peg Assembly",
      summary: z
        ? "仅靠原始画面与关节读数，抓起银色圆环并把内孔以 4.5 mm 精度套进圆柱销。"
        : "Picks up the round ring by its side tab, aligns the hole concentrically within 4.5 mm using only raw cameras, and seats it onto the peg.",
      badge: `${mm}mm`,
      badgeCls: "blue",
      resultLbl: z ? "同心对准误差" : "Alignment Error",
      resultVal: `${mm} mm`,
    };
  }
  if (tn === "Wipe") {
    const w = fm.wiped_markers ?? 40;
    const tot = fm.total_markers ?? 40;
    return {
      title: z ? "换装海绵垫表面擦除（UR5e）" : "Tool-Swapped Surface Wiping",
      summary: z
        ? "把机械臂末端从夹爪换成固定海绵垫，现场读取新 URDF 合成控制，轻触桌面擦净全部 40 处污渍。"
        : "Swapped the gripper for a fixed sponge pad, synthesized a fresh controller from the new URDF on the spot, and wiped all 40 marks clean.",
      badge: `${Math.round((w / Math.max(1, tot)) * 100)}%`,
      badgeCls: "blue",
      resultLbl: z ? "污渍擦除率" : "Surface Cleaned",
      resultVal: `${w} / ${tot} (100%)`,
    };
  }
  if (tn === "Door") {
    const deg = (fm.door_hinge_deg || 0).toFixed(1);
    return {
      title: z ? `拉把手开门（${d.robosuite_robot}）` : `Open Hinged Door (${d.robosuite_robot})`,
      summary: z
        ? `在未见过 ${d.robosuite_robot} 裸 URDF 上旋转手腕勾住门把手，沿门轴弧线顺畅拉开 ${deg}°。`
        : `Zero-shot on unseen ${d.robosuite_robot}: rotates wrist to hook behind the handle and pulls the hinged door open to ${deg}°.`,
      badge: `${deg}°`,
      badgeCls: "blue",
      resultLbl: z ? "门轴开角" : "Door Opened",
      resultVal: `${deg}°`,
    };
  }
  if (tn.startsWith("PickPlace")) {
    const mm = ((fm.xy_alignment_err_m || 0) * 1000).toFixed(1);
    return {
      title: z ? `跨围挡牛奶分拣（${d.robosuite_robot}）` : `Sort Milk Over Walls (${d.robosuite_robot})`,
      summary: z
        ? `使用未见过 ${d.robosuite_robot} 抓起牛奶盒，高举越过木箱隔板并稳稳放进对应分拣格。`
        : `Unseen ${d.robosuite_robot} grasps the milk carton, lifts it cleanly over the wooden dividers, and drops it into the matching slot.`,
      badge: `${mm}mm`,
      badgeCls: "blue",
      resultLbl: z ? "落点偏差" : "Slot Accuracy",
      resultVal: `${mm} mm`,
    };
  }
  const mm = ((fm.xy_alignment_err_m || 0) * 1000).toFixed(1);
  return {
    title: z ? `3D 方块精准堆叠（${d.robosuite_robot}）` : `3D Block Stacking (${d.robosuite_robot})`,
    summary: z
      ? `未见过 ${d.robosuite_robot} 抓起红方块并精准叠放到绿方块正上方（偏差仅 ${mm} mm）。`
      : `Unseen ${d.robosuite_robot} picks up the red block and stacks it squarely on top of the green block within ${mm} mm.`,
    badge: `${mm}mm`,
    badgeCls: "blue",
    resultLbl: z ? "堆叠偏差" : "Stack Accuracy",
    resultVal: `${mm} mm`,
  };
}

/* ── Live 60fps Interactive Hero Robot Arm Sandbox ────────────────────── */
const HERO_SIM = {
  targetX: 0.54,
  targetY: 0.44,
  curX: 0.54,
  curY: 0.44,
  gripClosed: false,
  cubeX: 0.64,
  cubeY: 0.735,
  holding: false,
  trail: [],
  autoTick: 0,
  probePhase: 0,
  probeRings: [],
};

const AUTO_SCRIPT = [
  { cmd: "MV_FWD", tx: 0.64, ty: 0.44, grip: false },
  { cmd: "MV_DOWN", tx: 0.64, ty: 0.69, grip: false },
  { cmd: "GRASP", tx: 0.64, ty: 0.69, grip: true },
  { cmd: "MV_UP", tx: 0.62, ty: 0.36, grip: true },
  { cmd: "MV_BACK", tx: 0.40, ty: 0.36, grip: true },
  { cmd: "MV_DOWN", tx: 0.40, ty: 0.69, grip: true },
  { cmd: "RELEASE", tx: 0.40, ty: 0.69, grip: false },
  { cmd: "MV_UP", tx: 0.44, ty: 0.42, grip: false },
  { cmd: "MV_BACK", tx: 0.40, ty: 0.44, grip: false },
  { cmd: "MV_DOWN", tx: 0.40, ty: 0.69, grip: false },
  { cmd: "GRASP", tx: 0.40, ty: 0.69, grip: true },
  { cmd: "MV_UP", tx: 0.48, ty: 0.36, grip: true },
  { cmd: "MV_FWD", tx: 0.64, ty: 0.38, grip: true },
  { cmd: "MV_DOWN", tx: 0.64, ty: 0.69, grip: true },
  { cmd: "RELEASE", tx: 0.64, ty: 0.69, grip: false },
  { cmd: "MV_UP", tx: 0.54, ty: 0.42, grip: false },
];

function flashHeroToken(tok) {
  const el = document.getElementById("hero-live-token");
  if (!el) return;
  el.textContent = tok;
  el.classList.remove("pulse");
  void el.offsetWidth;
  el.classList.add("pulse");
  setTimeout(() => el.classList.remove("pulse"), 320);
}

function updateHeroCaption() {
  const cap = document.getElementById("hero-canvas-caption");
  if (!cap) return;
  const z = STATE.lang === "zh";
  if (STATE.simMode === "blind") {
    cap.textContent = z
      ? "无 URDF 模式：轻推关节、观察指尖像素位移，几秒内在线估出运动规律。"
      : "No-URDF mode: nudging joints and watching visual fingertip shift to estimate motion live.";
  } else if (STATE.simMode === "urdf7") {
    cap.textContent = z
      ? "7 轴冗余协作臂：读取裸 URDF 后自动执行同一套通用技能命令。"
      : "7-DoF redundant cobot: executes the exact same universal skill commands from its raw URDF.";
  } else {
    cap.textContent = z
      ? "6 轴工业臂：点击下方任意按钮，实时体验同一套通用技能如何驱动机械臂。"
      : "6-DoF industrial arm: click any button below to drive the arm with shared universal skills.";
  }
}

function triggerHeroCmd(cmd) {
  if (cmd === "AUTO") {
    STATE.simAuto = !STATE.simAuto;
    document.getElementById("hero-auto-btn")?.classList.toggle("active", STATE.simAuto);
    return;
  }
  STATE.simAuto = false;
  document.getElementById("hero-auto-btn")?.classList.remove("active");

  const step = 0.085;
  if (cmd === "MV_FWD") HERO_SIM.targetX = Math.min(0.76, HERO_SIM.targetX + step);
  if (cmd === "MV_BACK") HERO_SIM.targetX = Math.max(0.26, HERO_SIM.targetX - step);
  if (cmd === "MV_UP") HERO_SIM.targetY = Math.max(0.22, HERO_SIM.targetY - step);
  if (cmd === "MV_DOWN") HERO_SIM.targetY = Math.min(0.70, HERO_SIM.targetY + step);
  if (cmd === "GRASP") {
    HERO_SIM.gripClosed = !HERO_SIM.gripClosed;
    const dist = Math.hypot(HERO_SIM.curX - HERO_SIM.cubeX, HERO_SIM.curY - (HERO_SIM.cubeY - 0.04));
    if (HERO_SIM.gripClosed && dist < 0.12) HERO_SIM.holding = true;
    if (!HERO_SIM.gripClosed) HERO_SIM.holding = false;
    flashHeroToken(HERO_SIM.gripClosed ? "GRASP" : "RELEASE");
    return;
  }
  flashHeroToken(cmd);
}

function solve2LinkIK(bx, by, tx, ty, l1, l2, flip = 1) {
  let dx = tx - bx;
  let dy = ty - by;
  const maxR = (l1 + l2) * 0.96;
  const minR = Math.abs(l1 - l2) + 8;
  let d = Math.hypot(dx, dy);
  if (d > maxR) {
    dx = (dx / d) * maxR;
    dy = (dy / d) * maxR;
    d = maxR;
  }
  if (d < minR) d = minR;
  const cosA = Math.max(-1, Math.min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)));
  const baseAng = Math.atan2(dy, dx);
  const a1 = baseAng - flip * Math.acos(cosA);
  const jx = bx + l1 * Math.cos(a1);
  const jy = by + l1 * Math.sin(a1);
  return { jx, jy, ex: bx + dx, ey: by + dy };
}

function startHeroCanvas() {
  const canvas = document.getElementById("hero-arm-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let lastAutoTime = performance.now();

  function frame(now) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.max(300, Math.round(rect.width));
    const H = Math.max(220, Math.round(rect.height));
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Auto choreography
    if (STATE.simAuto && now - lastAutoTime > 1250) {
      lastAutoTime = now;
      if (STATE.simMode === "blind") {
        HERO_SIM.probePhase = (HERO_SIM.probePhase + 1) % 6;
        const probes = [
          { tok: "WIGGLE_TCP", tx: 0.50, ty: 0.44, grip: true },
          { tok: "PROBE_J1", tx: 0.57, ty: 0.40, grip: false },
          { tok: "PROBE_J2", tx: 0.52, ty: 0.34, grip: false },
          { tok: "EST_J_VIS", tx: 0.64, ty: 0.50, grip: false },
          { tok: "MV_DOWN", tx: 0.64, ty: 0.69, grip: true },
          { tok: "MV_UP", tx: 0.62, ty: 0.38, grip: true },
        ];
        const p = probes[HERO_SIM.probePhase];
        HERO_SIM.targetX = p.tx;
        HERO_SIM.targetY = p.ty;
        HERO_SIM.gripClosed = p.grip;
        const nearCube = Math.hypot(HERO_SIM.curX - HERO_SIM.cubeX, HERO_SIM.curY - (HERO_SIM.cubeY - 0.045)) < 0.12;
        HERO_SIM.holding = p.grip && nearCube;
        if (!p.grip) HERO_SIM.holding = false;
        HERO_SIM.probeRings.push({ x: HERO_SIM.curX * W, y: HERO_SIM.curY * H, r: 6, a: 0.85 });
        flashHeroToken(p.tok);
      } else {
        const s = AUTO_SCRIPT[HERO_SIM.autoTick % AUTO_SCRIPT.length];
        HERO_SIM.autoTick += 1;
        HERO_SIM.targetX = s.tx;
        HERO_SIM.targetY = s.ty;
        HERO_SIM.gripClosed = s.grip;
        if (s.cmd === "GRASP" && Math.hypot(HERO_SIM.curX - HERO_SIM.cubeX, HERO_SIM.curY - (HERO_SIM.cubeY - 0.045)) < 0.13) {
          HERO_SIM.holding = true;
        }
        if (s.cmd === "RELEASE") HERO_SIM.holding = false;
        flashHeroToken(s.cmd);
      }
    }

    // Spring smooth interpolation
    HERO_SIM.curX += (HERO_SIM.targetX - HERO_SIM.curX) * 0.09;
    HERO_SIM.curY += (HERO_SIM.targetY - HERO_SIM.curY) * 0.09;

    const tableY = H * 0.78;
    const cubeRestY = (tableY - 13) / H;

    // Subtle paper drafting grid
    ctx.strokeStyle = "rgba(180, 188, 198, 0.24)";
    ctx.lineWidth = 1;
    for (let x = 24; x < W; x += 28) {
      ctx.beginPath();
      ctx.moveTo(x, 12);
      ctx.lineTo(x, tableY);
      ctx.stroke();
    }
    for (let y = 24; y < tableY; y += 28) {
      ctx.beginPath();
      ctx.moveTo(12, y);
      ctx.lineTo(W - 12, y);
      ctx.stroke();
    }

    // Table surface
    ctx.fillStyle = "#EFECE5";
    ctx.fillRect(24, tableY, W - 48, 8);
    ctx.strokeStyle = "#DCD6CC";
    ctx.beginPath();
    ctx.moveTo(24, tableY);
    ctx.lineTo(W - 24, tableY);
    ctx.stroke();

    // Target landing mat on left
    ctx.fillStyle = "rgba(62, 98, 128, 0.1)";
    ctx.fillRect(W * 0.33, tableY - 4, W * 0.14, 4);

    // Base pedestal
    const bx = W * 0.22;
    const by = tableY - 2;

    // Rigid arm kinematics (6-DoF 2-link vs 7-DoF true fixed-length 3-link vs Blind)
    const desiredWristX = HERO_SIM.curX * W;
    const desiredWristY = HERO_SIM.curY * H - 22;
    const shoulderX = bx;
    const shoulderY = by - (STATE.simMode === "urdf7" ? 24 : 18);
    const strokeMain = STATE.simMode === "urdf7" ? "#6A5484" : STATE.simMode === "blind" ? "#7A6A8E" : "#3E6280";
    const strokeSoft = STATE.simMode === "urdf7" ? "#B8ACC9" : STATE.simMode === "blind" ? "#C5B9D4" : "#9DB3C7";
    const scale = Math.max(265, Math.min(W * 0.62, H * 1.15));

    let joints = [];
    let actualWristX = desiredWristX;
    let actualWristY = desiredWristY;

    if (STATE.simMode === "urdf7") {
      // True fixed-length 3-link redundant cobot chain: Shoulder -> Elbow -> Forearm-Pitch -> Wrist
      const l1 = scale * 0.37;
      const l2 = scale * 0.35;
      const l3 = scale * 0.26;
      // Approach angle of link 3 into wrist adapts smoothly with horizontal reach
      const reachFrac = Math.max(0, Math.min(1, (desiredWristX - shoulderX) / (scale * 0.75)));
      const phi = 0.72 - reachFrac * 0.36; // ~41° down-right near base, ~21° at far reach
      const p3x = desiredWristX - l3 * Math.cos(phi);
      const p3y = desiredWristY - l3 * Math.sin(phi);
      const ik = solve2LinkIK(shoulderX, shoulderY, p3x, p3y, l1, l2, 1);
      actualWristX = ik.ex + l3 * Math.cos(phi);
      actualWristY = ik.ey + l3 * Math.sin(phi);
      joints = [
        [shoulderX, shoulderY, 7],
        [ik.jx, ik.jy, 6.5],
        [ik.ex, ik.ey, 5.8],
        [actualWristX, actualWristY, 4.8],
      ];
    } else {
      const l1 = scale * 0.50;
      const l2 = scale * 0.46;
      const ik = solve2LinkIK(shoulderX, shoulderY, desiredWristX, desiredWristY, l1, l2, 1);
      actualWristX = ik.ex;
      actualWristY = ik.ey;
      joints = [
        [shoulderX, shoulderY, 7],
        [ik.jx, ik.jy, 6.5],
        [actualWristX, actualWristY, 5],
      ];
    }

    const actualTipX = actualWristX;
    const actualTipY = actualWristY + 22;

    // Cube physics locked to actual solved end-effector position
    if (HERO_SIM.holding) {
      HERO_SIM.cubeX += (actualTipX / W - HERO_SIM.cubeX) * 0.35;
      HERO_SIM.cubeY += (actualTipY / H - HERO_SIM.cubeY) * 0.35;
    } else {
      HERO_SIM.cubeY += (cubeRestY - HERO_SIM.cubeY) * 0.18;
    }

    // Motion trail bound to actual solved fingertip
    HERO_SIM.trail.push({ x: actualTipX, y: actualTipY });
    if (HERO_SIM.trail.length > 26) HERO_SIM.trail.shift();
    if (HERO_SIM.trail.length > 2) {
      for (let i = 1; i < HERO_SIM.trail.length; i++) {
        const alpha = (i / HERO_SIM.trail.length) * 0.42;
        ctx.strokeStyle =
          STATE.simMode === "blind" || STATE.simMode === "urdf7"
            ? `rgba(106, 84, 132, ${alpha})`
            : `rgba(62, 98, 128, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(HERO_SIM.trail[i - 1].x, HERO_SIM.trail[i - 1].y);
        ctx.lineTo(HERO_SIM.trail[i].x, HERO_SIM.trail[i].y);
        ctx.stroke();
      }
    }

    // Probe sonar rings in blind mode
    HERO_SIM.probeRings = HERO_SIM.probeRings.filter((rg) => rg.a > 0.03);
    for (const rg of HERO_SIM.probeRings) {
      rg.r += 1.3;
      rg.a *= 0.94;
      ctx.strokeStyle = `rgba(106, 84, 132, ${rg.a})`;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(rg.x, rg.y, rg.r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw cube
    const cx = HERO_SIM.cubeX * W;
    const cy = HERO_SIM.cubeY * H;
    ctx.fillStyle = "#C7747C";
    ctx.strokeStyle = "#98505E";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(cx - 13, cy - 13, 26, 26, 5);
    ctx.fill();
    ctx.stroke();

    // Base pedestal
    ctx.fillStyle = "#2E343B";
    ctx.beginPath();
    ctx.roundRect(bx - 22, by - (STATE.simMode === "urdf7" ? 22 : 16), 44, STATE.simMode === "urdf7" ? 24 : 18, 5);
    ctx.fill();

    // Draw rigid arm links cleanly from joints array
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = 0; i < joints.length - 1; i++) {
      ctx.strokeStyle = i === 0 ? strokeMain : i === 1 ? strokeSoft : strokeMain;
      ctx.lineWidth = i === 0 ? 11 : i === 1 ? 8.8 : 6.8;
      ctx.beginPath();
      ctx.moveTo(joints[i][0], joints[i][1]);
      ctx.lineTo(joints[i + 1][0], joints[i + 1][1]);
      ctx.stroke();
    }

    // Wrist & parallel gripper (flush with cube sides at 13.5px when closed)
    ctx.strokeStyle = "#3B4248";
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(actualWristX, actualWristY);
    ctx.lineTo(actualWristX, actualWristY + 12);
    ctx.stroke();

    const jawGap = HERO_SIM.gripClosed ? 13.8 : 20.5;
    ctx.strokeStyle = "#2A2F34";
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.moveTo(actualWristX - jawGap, actualWristY + 12);
    ctx.lineTo(actualWristX + jawGap, actualWristY + 12);
    ctx.moveTo(actualWristX - jawGap, actualWristY + 12);
    ctx.lineTo(actualWristX - jawGap, actualWristY + 30);
    ctx.moveTo(actualWristX + jawGap, actualWristY + 12);
    ctx.lineTo(actualWristX + jawGap, actualWristY + 30);
    ctx.stroke();

    // Joint circles
    for (const [jx, jy, rad] of joints) {
      ctx.fillStyle = "#FFFFFF";
      ctx.strokeStyle = strokeMain;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(jx, jy, rad, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Visual-Motor Jacobian arrow hint in No-URDF mode
    if (STATE.simMode === "blind") {
      ctx.strokeStyle = "rgba(152, 80, 94, 0.72)";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(actualWristX, actualWristY + 24);
      ctx.lineTo(actualWristX + 32, actualWristY + 24);
      ctx.moveTo(actualWristX, actualWristY + 24);
      ctx.lineTo(actualWristX, actualWristY - 8);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#98505E";
      ctx.font = "600 10px ui-monospace, monospace";
      ctx.fillText("J_vis (+X, +Z)", actualWristX + 10, actualWristY + 38);
    }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* ── Mini 60fps Skill Preview Canvas (Chapter 2 Tab 1) ────────────────── */
let skillPreviewStop = null;
function startSkillPreviewCanvas(skillObj) {
  if (skillPreviewStop) {
    skillPreviewStop();
    skillPreviewStop = null;
  }
  const canvas = document.getElementById("skill-preview-canvas");
  if (!canvas || !skillObj) return;
  const ctx = canvas.getContext("2d");
  let alive = true;
  let t0 = performance.now();

  function draw(now) {
    if (!alive) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.max(240, Math.round(rect.width));
    const H = Math.max(180, Math.round(rect.height));
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const elapsed = (now - t0) / 1000;
    const osc = Math.sin(elapsed * 2.4);
    const prog = (elapsed * 0.65) % 1;
    const ease = prog < 0.5 ? 2 * prog * prog : 1 - Math.pow(-2 * prog + 2, 2) / 2;
    const cx = W / 2;
    const cy = H / 2;

    // Subtle crosshair
    ctx.strokeStyle = "#ECEAE4";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 16);
    ctx.lineTo(cx, H - 16);
    ctx.moveTo(24, cy);
    ctx.lineTo(W - 24, cy);
    ctx.stroke();

    if (skillObj.mode === "move") {
      const [vx, vy, vz] = skillObj.vec;
      const dx = (vx * 52 + vy * -42) * ease;
      const dy = (vz * -44 + vy * 18) * ease;
      // Ghost origin
      ctx.strokeStyle = "#CBD5E1";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(cx - dx * 0.45, cy - dy * 0.45, 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      // Arrow
      ctx.strokeStyle = "#3E6280";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx - dx * 0.45, cy - dy * 0.45);
      ctx.lineTo(cx + dx * 0.55, cy + dy * 0.55);
      ctx.stroke();
      // Active TCP + jaws
      const px = cx + dx * 0.55;
      const py = cy + dy * 0.55;
      ctx.fillStyle = "#3E6280";
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#2A2F34";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px - 14, py + 8);
      ctx.lineTo(px + 14, py + 8);
      ctx.moveTo(px - 14, py + 8);
      ctx.lineTo(px - 14, py + 22);
      ctx.moveTo(px + 14, py + 8);
      ctx.lineTo(px + 14, py + 22);
      ctx.stroke();
    } else if (skillObj.mode === "rot") {
      const angle = skillObj.dir * ease * 1.15;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.strokeStyle = "#DCD4E8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(0, 0, 48, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "#6A5484";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(0, 0, 48, -Math.PI / 2, -Math.PI / 2 + angle, skillObj.dir < 0);
      ctx.stroke();
      ctx.rotate(angle);
      ctx.fillStyle = "#6A5484";
      ctx.beginPath();
      ctx.roundRect(-24, -7, 48, 14, 6);
      ctx.fill();
      ctx.fillStyle = "#98505E";
      ctx.beginPath();
      ctx.arc(20, 0, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else if (skillObj.mode === "gear") {
      const stride = skillObj.size * 18;
      const x0 = cx - stride * 1.3;
      for (let i = 0; i <= 3; i++) {
        const px = x0 + i * stride * 0.85;
        ctx.fillStyle = i === Math.floor(prog * 4) ? "#3E6280" : "#CBD5E1";
        ctx.beginPath();
        ctx.arc(px, cy, i === Math.floor(prog * 4) ? 9 : 5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = "#3E6280";
      ctx.font = "700 13px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(`step = ${skillObj.size}.0 cm`, cx, cy + 44);
    } else if (skillObj.mode === "grip") {
      const gap = skillObj.closed ? 28 - ease * 17 : 11 + ease * 17;
      ctx.fillStyle = "#C7747C";
      ctx.beginPath();
      ctx.roundRect(cx - 10, cy - 2, 20, 26, 4);
      ctx.fill();
      ctx.strokeStyle = "#2A2F34";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(cx - gap, cy - 18);
      ctx.lineTo(cx + gap, cy - 18);
      ctx.moveTo(cx - gap, cy - 18);
      ctx.lineTo(cx - gap, cy + 20);
      ctx.moveTo(cx + gap, cy - 18);
      ctx.lineTo(cx + gap, cy + 20);
      ctx.stroke();
    } else if (skillObj.mode === "base") {
      const [bx, by, byaw] = skillObj.vec;
      ctx.save();
      ctx.translate(cx + bx * ease * 42, cy - by * ease * 36);
      ctx.rotate(-byaw * ease * 0.85);
      ctx.fillStyle = "#F0F4F8";
      ctx.strokeStyle = "#3E6280";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(-28, -18, 56, 36, 10);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#98505E";
      ctx.beginPath();
      ctx.arc(16, 0, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else {
      const r = 18 + Math.abs(osc) * 8;
      ctx.strokeStyle = "#6A5484";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "#3E6280";
      ctx.font = "700 13px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(skillObj.id, cx, cy + 5);
    }

    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  skillPreviewStop = () => {
    alive = false;
  };
}

/* ── Chapter 1: Real 3D Demos ─────────────────────────────────────────── */
let tokenReplayTimer = null;
function playTokenRipple() {
  if (tokenReplayTimer) clearInterval(tokenReplayTimer);
  const chips = Array.from(document.querySelectorAll("#active-token-stream .skill-tok"));
  if (!chips.length) return;
  let i = 0;
  chips.forEach((c) => c.classList.remove("lit"));
  chips[0].classList.add("lit");
  tokenReplayTimer = setInterval(() => {
    chips.forEach((c) => c.classList.remove("lit"));
    i += 1;
    if (i >= chips.length) {
      clearInterval(tokenReplayTimer);
      tokenReplayTimer = null;
      return;
    }
    chips[i].classList.add("lit");
  }, 420);
}

function renderDemos() {
  const z = STATE.lang === "zh";
  const demos = SNAP.mujoco_unseen_demos || [];
  if (!demos.length) return;

  const filterDefs = [
    ["all", z ? `全部 (${demos.length})` : `All (${demos.length})`],
    ["urdf_free", z ? "无 URDF 自试探" : "No URDF"],
    ["bimanual", z ? "异构双臂" : "Two Different Arms"],
    ["wipe", z ? "换工具擦除" : "Surface Wiping"],
    ["nut_assembly", z ? "精密套环" : "Precision Assembly"],
    ["door", z ? "拉门" : "Doors"],
    ["pick_place", z ? "分拣" : "Sorting"],
    ["stack", z ? "堆叠" : "Stacking"],
  ];

  const available = filterDefs.filter(
    ([k]) => k === "all" || demos.some((d) => d.task_family_key === k)
  );

  document.getElementById("task-filters").innerHTML = available
    .map(
      ([k, label]) => `
      <button type="button" class="filter-btn ${STATE.taskFilter === k ? "active" : ""}" data-tfilter="${k}">
        ${esc(label)}
      </button>`
    )
    .join("");

  document.querySelectorAll("[data-tfilter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      STATE.taskFilter = btn.getAttribute("data-tfilter");
      const filtered =
        STATE.taskFilter === "all"
          ? demos
          : demos.filter((d) => d.task_family_key === STATE.taskFilter);
      if (filtered.length && !filtered.some((d) => d.run_id === STATE.selectedRunId)) {
        STATE.selectedRunId = filtered[0].run_id;
        STATE.openTurn = 1;
      }
      renderDemos();
    });
  });

  const visible =
    STATE.taskFilter === "all"
      ? demos
      : demos.filter((d) => d.task_family_key === STATE.taskFilter);
  if (!STATE.selectedRunId || !visible.some((d) => d.run_id === STATE.selectedRunId)) {
    STATE.selectedRunId = visible[0].run_id;
    STATE.openTurn = 1;
  }
  const cur = visible.find((d) => d.run_id === STATE.selectedRunId) || visible[0];
  const meta = getDemoPlainMeta(cur, z);
  const turns = cur.stage2_vlm_closed_loop_turns || [];
  const turnIdx = Math.max(1, Math.min(STATE.openTurn, turns.length || 1));
  STATE.openTurn = turnIdx;
  const activeTurn = turns[turnIdx - 1] || {};
  const s1 = cur.stage1_urdf_and_synthesis || {};

  // Build universal skill chips for this turn
  let chipsHtml = "";
  if (cur.is_bimanual && (activeTurn.left_chunk || activeTurn.right_chunk)) {
    const lChips = (activeTurn.left_chunk || []).map((u) => `<span class="skill-tok left">L:${esc(u)}</span>`).join("");
    const rChips = (activeTurn.right_chunk || []).map((u) => `<span class="skill-tok right">R:${esc(u)}</span>`).join("");
    chipsHtml = `${lChips}${rChips}`;
  } else {
    chipsHtml = (activeTurn.action_chunk || [])
      .map((u) => {
        const sp = u === "GRASP" || u === "RELEASE" || u === "DONE" || u.startsWith("STEP_");
        return `<span class="skill-tok ${sp ? "special" : ""}">${esc(u)}</span>`;
      })
      .join("");
  }

  const filmstripHtml = turns
    .map(
      (t) => `
      <button type="button" class="film-thumb ${t.turn === turnIdx ? "active" : ""}" data-jump-turn="${t.turn}">
        <img src="./${esc(t.observation_image)}" alt="Turn ${t.turn}" loading="lazy" />
        <span>T${t.turn}</span>
      </button>`
    )
    .join("");

  document.getElementById("mujoco-theater-body").innerHTML = `
    <div class="theater-left">
      <div class="demo-header-row">
        <span class="badge ${meta.badgeCls}">${esc(meta.title)}</span>
        <span class="badge oat">${esc(cur.model)} · ${cur.total_vlm_turns} ${z ? "步" : "turns"}</span>
      </div>
      <p class="demo-summary">${esc(meta.summary)}</p>
      <div class="video-frame">
        <video src="./${esc(cur.video_path)}" controls autoplay loop muted playsinline></video>
      </div>
      <div class="filmstrip">${filmstripHtml}</div>
    </div>

    <div class="theater-right">
      <div class="result-bar">
        <div class="result-chip">
          <span>${esc(meta.resultLbl)}</span>
          <strong>${esc(meta.resultVal)}</strong>
        </div>
        <div class="result-chip">
          <span>${z ? "测试条件" : "Fairness"}</span>
          <strong>${cur.urdf_free ? (z ? "零 URDF · 零作弊" : "No URDF · Zero Cheats") : (z ? "裸 URDF · 零作弊" : "Raw URDF · Zero Cheats")}</strong>
        </div>
      </div>

      <div class="step-card">
        <div class="step-nav">
          <span class="step-nav-title">${z ? `第 ${turnIdx} / ${turns.length} 步 · AI 思考与决策` : `Step ${turnIdx} / ${turns.length} · What the AI Thought`}</span>
          <div class="step-btns">
            <button type="button" class="mini-btn" id="btn-prev-turn" ${turnIdx <= 1 ? "disabled" : ""}>←</button>
            <button type="button" class="mini-btn" id="btn-next-turn" ${turnIdx >= turns.length ? "disabled" : ""}>→</button>
          </div>
        </div>
        ${activeTurn.observation_image ? `<img class="step-img" src="./${esc(activeTurn.observation_image)}" alt="Step ${turnIdx}" />` : ""}
        <p class="thought-quote">“${esc(activeTurn.thought || "")}”</p>
        <div class="token-stream-head">
          <span>${z ? "发出的通用技能指令" : "Universal Skills Fired"}</span>
          <button type="button" class="mini-btn" id="btn-replay-tokens">▶ ${z ? "动效回放" : "Animate"}</button>
        </div>
        <div class="token-stream" id="active-token-stream">${chipsHtml}</div>
      </div>

      <details class="tech-drawer">
        <summary>
          <span>${z ? "展开技术细节（完整 Prompt / URDF / 合成代码）" : "Inspect technical details (Prompt, URDF & Code)"}</span>
          <span>+</span>
        </summary>
        <div class="tech-drawer-body">
          ${cur.self_probe_montage ? `<img class="step-img" src="./${esc(cur.self_probe_montage)}" alt="Self-probing montage" />` : ""}
          <div style="font-size:12px;font-weight:700;color:var(--muted)">${z ? "当前步完整零特权 Prompt：" : "Full Zero-Privilege Prompt (this step):"}</div>
          <pre class="code-box">${esc(activeTurn.prompt || "")}</pre>
          <div style="font-size:12px;font-weight:700;color:var(--muted)">${z ? "输入给 AI 的裸 URDF / 自试探报告：" : "Input Raw URDF / Self-Probe Model:"}</div>
          <pre class="code-box">${esc(s1.urdf_xml || "")}</pre>
          ${s1.synthesized_interpreter_code ? `<div style="font-size:12px;font-weight:700;color:var(--muted)">${z ? "AI 自动编写的 Python 控制代码：" : "AI-Written Python Controller:"}</div><pre class="code-box">${esc(s1.synthesized_interpreter_code)}</pre>` : ""}
        </div>
      </details>
    </div>
  `;

  document.getElementById("btn-prev-turn")?.addEventListener("click", () => {
    if (STATE.openTurn > 1) {
      STATE.openTurn -= 1;
      renderDemos();
    }
  });
  document.getElementById("btn-next-turn")?.addEventListener("click", () => {
    if (STATE.openTurn < turns.length) {
      STATE.openTurn += 1;
      renderDemos();
    }
  });
  document.getElementById("btn-replay-tokens")?.addEventListener("click", playTokenRipple);
  document.querySelectorAll("[data-jump-turn]").forEach((btn) => {
    btn.addEventListener("click", () => {
      STATE.openTurn = Number(btn.getAttribute("data-jump-turn"));
      renderDemos();
    });
  });

  // Selector cards underneath
  document.getElementById("task-matrix").innerHTML = visible
    .map((d) => {
      const m = getDemoPlainMeta(d, z);
      return `
        <button type="button" class="demo-tile ${d.run_id === cur.run_id ? "active" : ""}" data-run-id="${esc(d.run_id)}">
          <div class="demo-tile-top">
            <span class="demo-tile-title">${esc(m.title)}</span>
            <span class="demo-tile-metric ${m.badgeCls}">${esc(m.badge)}</span>
          </div>
          <span class="demo-tile-sub">${d.total_vlm_turns} ${z ? "步" : "turns"} · ${d.video_duration_s}s · ${esc(d.model)}</span>
        </button>`;
    })
    .join("");

  document.querySelectorAll("[data-run-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      STATE.selectedRunId = btn.getAttribute("data-run-id");
      STATE.openTurn = 1;
      renderDemos();
    });
  });

  playTokenRipple();
}

/* ── Chapter 2: Universal Skill Library ───────────────────────────────── */
function renderSkills() {
  const z = STATE.lang === "zh";
  const uSkills = SNAP.universal_skills || {};
  const tabs = [
    ["motion", z ? "25 个基础动作技能" : "25 Motion Skills"],
    ["engine", z ? `跨机身共享引擎 (${uSkills.common_base_loc || 244} 行)` : `Shared Engine (${uSkills.common_base_loc || 244} LOC)`],
    ["habits", z ? "操作经验库 & 114 条跨机轨迹" : "Manipulation Habits & Priors"],
    ["probe", z ? "无 URDF 自主试探技能" : "No-URDF Self-Discovery"],
  ];

  document.getElementById("skill-tabs").innerHTML = tabs
    .map(
      ([k, lbl]) => `
      <button type="button" class="skill-tab-btn ${STATE.skillTab === k ? "active" : ""}" data-stab="${k}">
        ${esc(lbl)}
      </button>`
    )
    .join("");

  document.querySelectorAll("[data-stab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      STATE.skillTab = btn.getAttribute("data-stab");
      renderSkills();
    });
  });

  const stage = document.getElementById("skill-stage");
  if (skillPreviewStop) {
    skillPreviewStop();
    skillPreviewStop = null;
  }

  if (STATE.skillTab === "motion") {
    const allItems = SKILL_GROUPS.flatMap((g) => g.items);
    const activeObj = allItems.find((x) => x.id === STATE.selectedSkill) || allItems[0];
    stage.innerHTML = `
      <div class="skills-motion-grid">
        <div class="skill-groups">
          ${SKILL_GROUPS.map(
            (g) => `
            <div class="skill-group">
              <h4>
                <span>${esc(z ? g.titleZh : g.titleEn)}</span>
                <span style="font-weight:500">${esc(z ? g.noteZh : g.noteEn)}</span>
              </h4>
              <div class="skill-chip-row">
                ${g.items
                  .map(
                    (it) => `
                    <button type="button" class="skill-btn ${it.id === activeObj.id ? "active" : ""}" data-skill-id="${it.id}">
                      ${it.id}
                    </button>`
                  )
                  .join("")}
              </div>
            </div>`
          ).join("")}
        </div>
        <div class="skill-preview-card">
          <canvas id="skill-preview-canvas" aria-label="Skill motion preview"></canvas>
          <h3>${esc(activeObj.id)}</h3>
          <p>${esc(z ? activeObj.zh : activeObj.en)}</p>
          <span style="font-size:12px;color:var(--muted)">
            ${z ? "点击左侧任意技能按钮，右侧即时动态预览对应动作效果。" : "Click any skill on the left to preview how every robot interprets it."}
          </span>
        </div>
      </div>`;
    document.querySelectorAll("[data-skill-id]").forEach((b) => {
      b.addEventListener("click", () => {
        STATE.selectedSkill = b.getAttribute("data-skill-id");
        renderSkills();
      });
    });
    startSkillPreviewCanvas(activeObj);
    return;
  }

  if (STATE.skillTab === "engine") {
    const reusers = uSkills.common_base_reusers || [];
    const code = SNAP.common_library?.code || "";
    stage.innerHTML = `
      <div class="engine-flow">
        <span class="flow-box">${z ? "任意新机器人裸 URDF" : "Any New Raw URDF"}</span>
        <span class="flow-sep">→</span>
        <span class="flow-box accent">GenericURDFInterpreter (${uSkills.common_base_loc || 244} ${z ? "行" : "lines"})</span>
        <span class="flow-sep">→</span>
        <span class="flow-box">${reusers.length} / ${(SNAP.robots || []).length} ${z ? "款机器人直接复用" : "Robots Reuse It"}</span>
      </div>
      <p style="font-size:14.5px;color:var(--ink);line-height:1.68">
        ${
          z
            ? "AI 在看懂几台机器人后，自动把所有机身共用的正逆运动学、6 轴平滑插值、关节限位保护与工作空间安全围栏抽成了一份 244 行通用基类。之后每接入一台新机器人，只需几行配置就能直接复用全部技能。"
            : "After studying the first few robots, the LLM abstracted shared FK/IK, smooth 6D motion, joint limits, and safety bounds into a single 244-line Python base class. Every new robot reuses it with just a thin declarative config."
        }
      </p>
      <div class="reuser-chips">
        ${reusers
          .map((id) => `<button type="button" class="reuser-chip" data-open-robot="${esc(id)}">${esc(id)} →</button>`)
          .join("")}
      </div>
      <div style="font-size:12px;font-weight:700;color:var(--muted);margin-bottom:6px">
        harnesses/common/urdf_interpreter_base.py
      </div>
      <pre class="code-box">${esc(code)}</pre>
    `;
    stage.querySelectorAll("[data-open-robot]").forEach((b) => {
      b.addEventListener("click", () => openRobotModal(b.getAttribute("data-open-robot")));
    });
    return;
  }

  if (STATE.skillTab === "habits") {
    const exemplars = SNAP.priors?.exemplars || [];
    const q = STATE.priorQuery.trim().toLowerCase();
    const matched = exemplars
      .filter((e) => !q || `${e.task_instruction} ${e.source_dataset || e.dataset || ""} ${e.source_embodiment || e.embodiment || ""}`.toLowerCase().includes(q))
      .slice(0, 6);

    stage.innerHTML = `
      <div class="habit-grid">
        ${HABIT_CARDS.map(
          (h) => `
          <article class="habit-card">
            <span class="badge purple" style="align-self:flex-start">${esc(z ? h.tagZh : h.tag)}</span>
            <h4>${esc(z ? h.titleZh : h.titleEn)}</h4>
            <p>${esc(z ? h.bodyZh : h.bodyEn)}</p>
          </article>`
        ).join("")}
      </div>
      <div class="prior-search-row">
        <div>
          <strong style="font-size:14.5px;color:var(--ink-strong)">
            ${z ? `跨机身真实示教检索（8 套开源数据集 · ${uSkills.retargeted_episodes || 114} 条轨迹）` : `Search Cross-Robot Skill Traces (${uSkills.datasets_covered || 8} Datasets · ${uSkills.retargeted_episodes || 114} Episodes)`}
          </strong>
        </div>
        <input type="search" class="search-input" id="prior-search-input" value="${esc(STATE.priorQuery)}" placeholder="${z ? "搜关键词：mug / bottle / drawer ..." : "Filter: mug / bottle / drawer ..."}" />
      </div>
      <div class="prior-list">
        ${matched
          .map(
            (e) => `
          <div class="prior-item">
            <div class="prior-meta">
              <span>${esc(e.source_dataset || e.dataset || "dataset")} · ${esc(e.source_embodiment || e.embodiment || "robot")}</span>
              <span>${e.num_units || (e.semantic_units || []).length} ${z ? "步" : "steps"}</span>
            </div>
            <div class="prior-task">${esc(e.task_instruction)}</div>
            <div class="token-stream">
              ${(e.semantic_units || []).slice(0, 8).map((u) => `<span class="skill-tok">${esc(u)}</span>`).join("")}
            </div>
          </div>`
          )
          .join("")}
      </div>
    `;
    const inp = document.getElementById("prior-search-input");
    if (inp) {
      inp.addEventListener("input", () => {
        STATE.priorQuery = inp.value;
        renderSkills();
        const nextInp = document.getElementById("prior-search-input");
        if (nextInp) {
          nextInp.focus();
          nextInp.setSelectionRange(nextInp.value.length, nextInp.value.length);
        }
      });
    }
    return;
  }

  // Tab 4: Blind No-URDF Self-Discovery Skill
  const blindDemo = (SNAP.mujoco_unseen_demos || []).find((d) => d.urdf_free);
  const steps = z
    ? [
        ["抖一下手指，找到自己的手在哪", "静止状态下开合夹爪一次，用原始画面像素差分直接锁定末端位置。"],
        ["逐一轻推每个关节，看画面怎么动", "对每个未知关节施加微小安全试探，记录每个关节移动带来的画面像素变化。"],
        ["估出视觉运动映射 J_vis", "无需任何 URDF 或 3D 坐标，直接解出“想往哪走 → 动哪些关节”的控制矩阵。"],
        ["边干活边用 Broyden 在线校准", "任务执行中持续跟踪末端位移并修正映射，12 轮内稳稳完成 3D 抓取举升。"],
      ]
    : [
        ["Wiggle fingers to spot its own hand", "Opens and closes the gripper once while still, using pixel difference across cameras to locate the fingertips."],
        ["Nudge each joint safely", "Applies tiny safe probes to each unknown joint and watches how the fingertips move in the camera views."],
        ["Estimate visual-motor map J_vis", "Solves which joint nudges move the hand forward, left, and up — with zero URDF and zero 3D sensors."],
        ["Adapt live with Broyden updates", "Refines the motion map online during real grasping, lifting the 3D cube cleanly in 12 turns."],
      ];

  stage.innerHTML = `
    <div class="probe-grid">
      <div class="probe-steps">
        ${steps
          .map(
            ([t, d], idx) => `
          <div class="probe-step">
            <span class="probe-num">${idx + 1}</span>
            <div>
              <strong>${esc(t)}</strong>
              <p>${esc(d)}</p>
            </div>
          </div>`
          )
          .join("")}
        ${
          blindDemo
            ? `<button type="button" class="btn btn-primary" id="jump-blind-demo" style="align-self:flex-start;margin-top:6px">
                ${z ? "看无 URDF 实拍 3D 视频 ↑" : "Watch the No-URDF 3D Video ↑"}
               </button>`
            : ""
        }
      </div>
      <div class="probe-visual">
        ${
          blindDemo?.self_probe_montage
            ? `<img src="./${esc(blindDemo.self_probe_montage)}" alt="URDF-free active self-probing montage" />`
            : ""
        }
      </div>
    </div>
  `;
  document.getElementById("jump-blind-demo")?.addEventListener("click", () => {
    if (!blindDemo) return;
    STATE.taskFilter = "all";
    STATE.selectedRunId = blindDemo.run_id;
    STATE.openTurn = 1;
    renderDemos();
    document.getElementById("demos")?.scrollIntoView({ behavior: "smooth" });
  });
}

/* ── Chapter 3: 15 Supported Robots ───────────────────────────────────── */
function robotSvg(r) {
  const stroke = r.split === "unseen" ? "#98505E" : r.is_mobile ? "#6A5484" : "#3E6280";
  const soft = r.split === "unseen" ? "#FBF2F3" : r.is_mobile ? "#F4F1F8" : "#F0F4F8";
  if (r.is_mobile && r.is_bimanual) {
    return `<svg viewBox="0 0 220 72"><rect x="76" y="48" width="68" height="14" rx="5" fill="${soft}" stroke="${stroke}" stroke-width="2"/><circle cx="90" cy="65" r="3.5" fill="${stroke}"/><circle cx="130" cy="65" r="3.5" fill="${stroke}"/><line x1="110" y1="48" x2="110" y2="24" stroke="${stroke}" stroke-width="2.8"/><path d="M110 28 L78 18 L56 30" fill="none" stroke="${stroke}" stroke-width="2.5" stroke-linecap="round"/><path d="M110 28 L142 18 L164 30" fill="none" stroke="${stroke}" stroke-width="2.5" stroke-linecap="round"/><circle cx="56" cy="30" r="3.5" fill="#C7747C"/><circle cx="164" cy="30" r="3.5" fill="#C7747C"/></svg>`;
  }
  if (r.is_mobile) {
    return `<svg viewBox="0 0 220 72"><rect x="64" y="48" width="66" height="14" rx="5" fill="${soft}" stroke="${stroke}" stroke-width="2"/><circle cx="78" cy="65" r="3.5" fill="${stroke}"/><circle cx="116" cy="65" r="3.5" fill="${stroke}"/><line x1="96" y1="48" x2="96" y2="24" stroke="${stroke}" stroke-width="3"/><path d="M96 24 L126 12 L154 26 L166 34" fill="none" stroke="${stroke}" stroke-width="2.6" stroke-linecap="round"/><circle cx="166" cy="34" r="3.5" fill="#C7747C"/></svg>`;
  }
  if (r.is_bimanual) {
    return `<svg viewBox="0 0 220 72"><line x1="45" y1="62" x2="175" y2="62" stroke="#D8DDE4" stroke-width="2"/><rect x="68" y="52" width="18" height="10" rx="2" fill="${soft}" stroke="${stroke}" stroke-width="1.8"/><rect x="134" y="52" width="18" height="10" rx="2" fill="${soft}" stroke="${stroke}" stroke-width="1.8"/><path d="M77 52 L64 30 L86 16 L98 24" fill="none" stroke="${stroke}" stroke-width="2.6" stroke-linecap="round"/><path d="M143 52 L156 30 L134 16 L122 24" fill="none" stroke="${stroke}" stroke-width="2.6" stroke-linecap="round"/></svg>`;
  }
  return `<svg viewBox="0 0 220 72"><line x1="55" y1="62" x2="165" y2="62" stroke="#D8DDE4" stroke-width="2"/><rect x="78" y="50" width="22" height="12" rx="3" fill="${soft}" stroke="${stroke}" stroke-width="2"/><path d="M89 50 L89 32 L118 14 L146 26 L154 38" fill="none" stroke="${stroke}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="89" cy="32" r="3" fill="#FFFFFF" stroke="${stroke}" stroke-width="2"/><circle cx="118" cy="14" r="3" fill="#FFFFFF" stroke="${stroke}" stroke-width="2"/><circle cx="154" cy="38" r="3.6" fill="#C7747C"/></svg>`;
}

function renderRobots() {
  const z = STATE.lang === "zh";
  const robots = SNAP.robots || [];
  const filters = [
    ["all", z ? `全部 (${robots.length})` : `All (${robots.length})`],
    ["unseen", z ? `未见裸 URDF (${robots.filter((r) => r.split === "unseen").length})` : `Unseen Raw URDF (${robots.filter((r) => r.split === "unseen").length})`],
    ["common", z ? `复用通用引擎 (${robots.filter((r) => r.used_common_library).length})` : `Shared Engine (${robots.filter((r) => r.used_common_library).length})`],
  ];

  document.getElementById("robot-filters").innerHTML = filters
    .map(
      ([k, lbl]) => `
      <button type="button" class="filter-btn ${STATE.robotFilter === k ? "active" : ""}" data-rfilter="${k}">
        ${esc(lbl)}
      </button>`
    )
    .join("");

  document.querySelectorAll("[data-rfilter]").forEach((b) => {
    b.addEventListener("click", () => {
      STATE.robotFilter = b.getAttribute("data-rfilter");
      renderRobots();
    });
  });

  const shown = robots.filter((r) => {
    if (STATE.robotFilter === "unseen") return r.split === "unseen";
    if (STATE.robotFilter === "common") return r.used_common_library;
    return true;
  });

  document.getElementById("robot-grid").innerHTML = shown
    .map((r) => {
      const chains = Object.values(r.chains || {});
      const maxPos = Math.max(0, ...chains.map((c) => c.max_directional_err_mm || 0)).toFixed(2);
      const morph =
        r.is_mobile && r.is_bimanual
          ? (z ? `移动双臂 2×${r.dof_per_arm} 轴` : `Mobile Dual ${r.dof_per_arm}-DoF`)
          : r.is_mobile
          ? (z ? `移动单臂 ${r.dof_per_arm} 轴` : `Mobile ${r.dof_per_arm}-DoF`)
          : r.is_bimanual
          ? (z ? `双臂 2×${r.dof_per_arm} 轴` : `Dual-Arm 2×${r.dof_per_arm}-DoF`)
          : (z ? `单臂 ${r.dof_per_arm} 轴` : `Single-Arm ${r.dof_per_arm}-DoF`);

      return `
        <button type="button" class="robot-card" data-open-robot="${esc(r.robot_id)}">
          <div class="robot-card-top">
            <h4>${esc(r.robot_id)}</h4>
            <span class="badge ${r.split === "unseen" ? "rose" : "blue"}">
              ${r.split === "unseen" ? (z ? "未见裸 URDF" : "Unseen URDF") : (z ? "已见本体" : "Seen")}
            </span>
          </div>
          <div class="robot-svg-wrap">${robotSvg(r)}</div>
          <div class="robot-foot">
            <span>${esc(morph)} · ≤${maxPos}mm</span>
            <strong style="color:var(--blue-ink)">${z ? "看 URDF ↔ 代码 →" : "URDF ↔ Code →"}</strong>
          </div>
        </button>`;
    })
    .join("");

  document.querySelectorAll("#robot-grid [data-open-robot]").forEach((btn) => {
    btn.addEventListener("click", () => openRobotModal(btn.getAttribute("data-open-robot")));
  });
}

function openRobotModal(robotId) {
  const r = (SNAP.robots || []).find((x) => x.robot_id === robotId);
  if (!r) return;
  const z = STATE.lang === "zh";
  const modal = document.getElementById("robot-modal");
  document.getElementById("modal-content").innerHTML = `
    <p class="kicker">${z ? "隔离物理验算通过" : "VERIFIED IN ISOLATED SUBPROCESS"}</p>
    <h2 id="modal-robot-title" style="font-family:var(--mono);font-size:21px;margin-top:6px;color:var(--ink-strong)">
      ${esc(r.robot_id)}
    </h2>
    <p style="font-size:13.5px;color:var(--muted);margin-top:4px">
      ${z ? `代码行数：${r.lines_of_code} 行 · 尝试次数：${r.synthesis?.attempts || 1} 轮 · AST 安全审计：通过 ✓` : `${r.lines_of_code} lines · ${r.synthesis?.attempts || 1} attempt(s) · AST Security Check: PASS ✓`}
    </p>
    <div class="dual-code">
      <div>
        <div style="font-size:12px;font-weight:700;color:var(--muted);margin-bottom:6px">
          ${z ? `原始 URDF 说明书 (robots/${esc(r.robot_id)}/robot.urdf)` : `Raw URDF Manual (robots/${esc(r.robot_id)}/robot.urdf)`}
        </div>
        <pre class="code-box">${esc(r.urdf_preview || "")}</pre>
      </div>
      <div>
        <div style="font-size:12px;font-weight:700;color:var(--muted);margin-bottom:6px">
          ${z ? `AI 自动生成的 Python 控制器 (harnesses/embodiments/${esc(r.robot_id)}.py)` : `AI-Written Python Controller (harnesses/embodiments/${esc(r.robot_id)}.py)`}
        </div>
        <pre class="code-box">${esc(r.code_preview || "")}</pre>
      </div>
    </div>
  `;
  modal.hidden = false;
  document.getElementById("modal-close")?.focus({ preventScroll: true });
}

/* ── Chapter 4: Try It Yourself + Comparison Drawer ───────────────────── */
function renderQuickstartAndCompare() {
  const z = STATE.lang === "zh";
  const tabs = [
    [
      "verify",
      z ? "1. 安装并验算 15 款机器人" : "1. Install & Verify",
      `git clone https://github.com/Interactive-Active-AI/EAISys.git
cd EAISys
uv sync --extra dev

# Verify all 15 AI-written robot controllers in isolated subprocesses
uv run eaisys verify`,
    ],
    [
      "synth",
      z ? "2. 接入任意新 URDF" : "2. Add Any New URDF",
      `export OPENAI_API_KEY="your-key"

# Give any unseen .urdf — EAISys writes & verifies its controller automatically
uv run eaisys synthesize robots/ur5e/robot.urdf --provider openai --model gpt-5.4

# Or run with your own local vLLM server
uv run eaisys synthesize robots/ur5e/robot.urdf --provider vllm --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-VL-72B-Instruct`,
    ],
    [
      "play",
      z ? "3. 跑 3D 任务（有/无 URDF）" : "3. Run 3D Tasks",
      `uv sync --extra sim --extra llm

# Pillar 1: Two different unseen URDFs working together (Kinova3 + UR5e)
uv run --extra sim eaisys play-bimanual-mujoco --env TwoArmLift --left-robot Kinova3 --right-robot UR5e --provider openai --model gpt-5.4

# Pillar 2: ZERO URDF — self-probes its own joints first, then solves the task
uv run --extra sim eaisys play-urdf-free --robosuite-robot Panda --env Lift --provider openai --model gpt-5.4`,
    ],
    [
      "export",
      z ? "4. 网页遥操 & 导出微调数据" : "4. Web Teleop & Export",
      `# Hardware-free browser teleoperation on any URDF
uv run --extra sim eaisys gumi --robot ur5e_mujoco --env Stack --robosuite-robot UR5e --port 8766

# Export verified 3D episodes into LLaMA-Factory SFT or GRPO datasets
uv run eaisys export-sft --format llamafactory --out artifacts/posttrain/sft.jsonl
uv run eaisys export-sft --format grpo --out artifacts/posttrain/grpo.jsonl`,
    ],
  ];

  const active = tabs.find((t) => t[0] === STATE.cliTab) || tabs[0];
  document.getElementById("cli-panel").innerHTML = `
    <div class="cli-bar">
      <div class="cli-tabs">
        ${tabs
          .map(
            ([k, lbl]) => `
            <button type="button" class="cli-tab ${k === active[0] ? "active" : ""}" data-ctab="${k}">
              ${esc(lbl)}
            </button>`
          )
          .join("")}
      </div>
      <button type="button" class="mini-btn" id="copy-cli-btn">${z ? "复制命令" : "Copy"}</button>
    </div>
    <pre class="code-box">${esc(active[2])}</pre>
  `;

  document.querySelectorAll("[data-ctab]").forEach((b) => {
    b.addEventListener("click", () => {
      STATE.cliTab = b.getAttribute("data-ctab");
      renderQuickstartAndCompare();
    });
  });
  document.getElementById("copy-cli-btn")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(active[2]);
      const btn = document.getElementById("copy-cli-btn");
      if (btn) {
        btn.textContent = z ? "已复制 ✓" : "Copied ✓";
        setTimeout(() => (btn.textContent = z ? "复制命令" : "Copy"), 1500);
      }
    } catch {
      /* noop */
    }
  });

  // Comparison table inside collapsible drawer
  const rows = z
    ? [
        ["端到端 VLA（π0.5 / OpenVLA）", "机器人形态锁死在权重里", "换机器人要重采示教再微调", "通常限于单构型"],
        ["手写具身框架（Show-Harness）", "工程师逐台手写控制器", "接入新机器需人工写 IK 与限位", "需手写双臂适配"],
        ["单身体智能体（RobotUse / CaP-X）", "预设固定单臂接口", "不支持直接读裸 URDF 泛化", "以单臂桌面为主"],
        ["EAISys v0.3（本项目）", "读裸 URDF 自动写控制器 + 共享 244 行技能引擎", "有 URDF 零样本即用；无 URDF 自主试探几秒即用", "原生支持异构双臂（Kinova3 + UR5e）"],
      ]
    : [
        ["End-to-End VLAs (π0.5, OpenVLA)", "Body geometry locked in weights", "Collect new demos & fine-tune", "Fixed to training arm"],
        ["Handwritten Harnesses (Show-Harness)", "Engineers hand-code each robot", "Manual IK, limits & frames", "Manual dual-arm setup"],
        ["Single-Body Agents (RobotUse, CaP-X)", "Pre-built single-arm backend", "No raw-URDF synthesis", "Mostly single-arm"],
        ["EAISys v0.3 (Ours)", "Reads raw URDF + shared 244-line skill engine", "Zero-shot with URDF · few-shot self-probe without URDF", "Native heterogeneous bimanual (Kinova3 + UR5e)"],
      ];

  document.getElementById("compare-table-wrap").innerHTML = `
    <table class="compare-table">
      <thead>
        <tr>
          <th>${z ? "路线" : "Approach"}</th>
          <th>${z ? "如何控制机器人身体" : "How It Controls Hardware"}</th>
          <th>${z ? "换一台全新机器人怎么办" : "Adding a Brand-New Robot"}</th>
          <th>${z ? "异构双臂支持" : "Two Different Arms"}</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (r, idx) => `
          <tr class="${idx === 3 ? "ours" : ""}">
            <td><strong>${esc(r[0])}</strong></td>
            <td>${esc(r[1])}</td>
            <td>${esc(r[2])}</td>
            <td>${esc(r[3])}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
}

/* ── Stats Count-Up & Scroll Reveal Choreography ──────────────────────── */
function updateStatsText() {
  const st = SNAP.stats || {};
  const v = st.mujoco_unseen_videos ?? 12;
  const r = st.total_robots ?? 15;
  const s = st.semantic_action_units ?? 25;
  const elV = document.getElementById("stat-videos");
  const elR = document.getElementById("stat-robots");
  const elS = document.getElementById("stat-skills");
  if (elV) elV.textContent = `${v} / ${v}`;
  if (elR) elR.textContent = `${r}`;
  if (elS) elS.textContent = `${s}`;
}

function animateCounter(el, target, suffix = "") {
  const start = performance.now();
  const dur = 750;
  function tick(now) {
    const p = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = Math.round(eased * target);
    el.textContent = suffix ? `${val}${suffix}` : `${val}`;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function setupScrollEffects() {
  const bar = document.getElementById("scroll-progress");
  window.addEventListener(
    "scroll",
    () => {
      const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      const pct = Math.min(100, (window.scrollY / max) * 100);
      if (bar) bar.style.width = `${pct}%`;
    },
    { passive: true }
  );

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("in-view");
        if (e.target.classList.contains("stats-strip") && !e.target.dataset.counted) {
          e.target.dataset.counted = "1";
          const st = SNAP.stats || {};
          const v = st.mujoco_unseen_videos ?? 12;
          const r = st.total_robots ?? 15;
          const s = st.semantic_action_units ?? 25;
          const elV = document.getElementById("stat-videos");
          const elR = document.getElementById("stat-robots");
          const elS = document.getElementById("stat-skills");
          if (elV) animateCounter(elV, v, ` / ${v}`);
          if (elR) animateCounter(elR, r);
          if (elS) animateCounter(elS, s);
        }
        io.unobserve(e.target);
      }
    },
    { threshold: 0.08 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

function applyLanguage(lang) {
  STATE.lang = lang;
  const dict = COPY[lang] || COPY.en;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const k = el.getAttribute("data-i18n");
    if (dict[k]) el.textContent = dict[k];
  });
  const h1 = document.getElementById("hero-title");
  if (h1) h1.textContent = dict.hero_title;

  document.getElementById("lang-en")?.classList.toggle("active", lang === "en");
  document.getElementById("lang-zh")?.classList.toggle("active", lang === "zh");
  document.querySelector(".lang-switch")?.classList.toggle("is-zh", lang === "zh");

  updateHeroCaption();
  renderDemos();
  renderSkills();
  renderRobots();
  renderQuickstartAndCompare();
}

function boot() {
  updateStatsText();
  setupScrollEffects();
  startHeroCanvas();

  document.getElementById("lang-en")?.addEventListener("click", () => applyLanguage("en"));
  document.getElementById("lang-zh")?.addEventListener("click", () => applyLanguage("zh"));

  // Mobile nav
  const toggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");
  toggle?.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  navLinks?.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      navLinks.classList.remove("open");
      toggle?.setAttribute("aria-expanded", "false");
    })
  );

  // Hero sandbox controls
  document.querySelectorAll("[data-sim-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      STATE.simMode = btn.getAttribute("data-sim-mode");
      document.querySelectorAll("[data-sim-mode]").forEach((b) => b.classList.toggle("active", b === btn));
      HERO_SIM.trail = [];
      updateHeroCaption();
      flashHeroToken(STATE.simMode === "blind" ? "WIGGLE_TCP" : "URDF_READY");
    });
  });
  document.querySelectorAll("[data-sim-cmd]").forEach((btn) => {
    btn.addEventListener("click", () => triggerHeroCmd(btn.getAttribute("data-sim-cmd")));
  });

  // Modal close handlers
  const modal = document.getElementById("robot-modal");
  document.getElementById("modal-close")?.addEventListener("click", () => {
    if (modal) modal.hidden = true;
  });
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) modal.hidden = true;
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && !modal.hidden) modal.hidden = true;
  });

  applyLanguage("en");
}

boot();
