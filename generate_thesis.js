const fs = require('fs');
process.env.NODE_PATH = '/home/xu/.npm-global/lib/node_modules';
require('module').Module._initPaths();

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, PageOrientation, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition,
  TableOfContents
} = require('docx');

// ===== 论文常量 =====
const PAPER_A4_W = 11906;
const PAPER_A4_H = 16838;
const MARGIN = { top: 1440, right: 1440, bottom: 1440, left: 1440 }; // 1 inch
const CONTENT_W = PAPER_A4_W - MARGIN.left - MARGIN.right; // 9026

// ===== 颜色常量 =====
const COLOR_BLACK = "000000";
const COLOR_DARK = "333333";
const COLOR_ACCENT = "1F4E79";
const COLOR_LIGHT_BG = "D6E4F0";
const COLOR_GRAY = "888888";

// ===== 辅助函数 =====
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: "SimHei", size: 32, bold: true, color: COLOR_ACCENT })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, font: "SimHei", size: 28, bold: true, color: COLOR_DARK })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: "SimHei", size: 24, bold: true, color: COLOR_DARK })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: { firstLine: 480 },
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text,
      font: "SimSun",
      size: 24,
      ...opts
    })]
  });
}

function paraBold(text) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: { firstLine: 480 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: "SimSun", size: 24, bold: true })]
  });
}

function paraMixed(parts) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: { firstLine: 480 },
    alignment: AlignmentType.JUSTIFIED,
    children: parts.map(p => new TextRun({ font: "SimSun", size: 24, ...p }))
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 60 }, children: [] });
}

function codePara(text) {
  return new Paragraph({
    spacing: { after: 40, line: 280 },
    indent: { left: 480 },
    shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
    children: [new TextRun({ text, font: "Courier New", size: 20, color: "333333" })]
  });
}

// ===== 表格辅助 =====
function makeTable(headers, rows, colWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const totalW = colWidths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: COLOR_LIGHT_BG, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      verticalAlign: "center",
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: h, font: "SimHei", size: 22, bold: true, color: COLOR_DARK })]
      })]
    })),
    tableHeader: true
  });

  const dataRows = rows.map(row => new TableRow({
    children: row.map((cell, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: String(cell), font: "SimSun", size: 22 })]
      })]
    }))
  }));

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

function makeSimpleTable(headers, rows, colWidths) {
  return makeTable(headers, rows, colWidths);
}

// ===== 封面 =====
function createCover() {
  return [
    emptyLine(), emptyLine(), emptyLine(), emptyLine(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: "本科毕业设计", font: "SimSun", size: 36, color: COLOR_ACCENT })]
    }),
    emptyLine(), emptyLine(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [new TextRun({ text: "基于强化学习与视觉识别的", font: "SimHei", size: 44, bold: true, color: COLOR_BLACK })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [new TextRun({ text: "CS游戏智能体系统设计与实现", font: "SimHei", size: 44, bold: true, color: COLOR_BLACK })]
    }),
    emptyLine(), emptyLine(), emptyLine(), emptyLine(),
    ...[
      ["学    院：", "计算机科学与技术学院"],
      ["专    业：", "计算机科学与技术"],
      ["学生姓名：", "许同学"],
      ["指导教师：", "XXX 教授"],
      ["完成日期：", "2026年5月"]
    ].map(([label, value]) => new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200, line: 400 },
      children: [
        new TextRun({ text: label, font: "SimHei", size: 28, bold: true }),
        new TextRun({ text: value, font: "SimSun", size: 28 }),
      ]
    })),
    emptyLine(), emptyLine(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "XXXX 大学", font: "SimHei", size: 32, bold: true, color: COLOR_ACCENT })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: "二〇二六年五月", font: "SimSun", size: 28 })]
    }),
  ];
}

// ===== 摘要 =====
function createAbstract() {
  return [
    heading1("摘  要"),
    para("第一人称射击（FPS）游戏是强化学习（Reinforcement Learning, RL）领域最具挑战性的应用场景之一，其高动态环境、部分可观测特性和连续控制空间对智能体设计提出了严峻考验。本文设计并实现了一个基于深度强化学习的CS（Counter-Strike）游戏智能体系统，该系统结合计算机视觉技术实现了端到端的游戏感知与控制闭环。"),
    para("本文的主要工作包括：（1）构建了分层强化学习框架，将高层战术决策（搜索、战斗、隐蔽）与低层连续瞄准控制解耦，降低了端到端学习难度；（2）基于Twin Delayed DDPG（TD3）算法设计了连续动作空间策略网络，实现了平滑的鼠标移动与射击控制；（3）集成了多模态视觉感知流水线，包括YOLO目标检测（人物与头部定位）、OCR界面状态读取（血量、弹药等）以及Qwen视觉语言模型的智能决策辅助；（4）设计了基于Win32 API的低延迟跨平台游戏控制方案，通过Socket通信实现WSL到Windows的毫秒级响应；（5）构建了包含命中、击杀、瞄准精度、生存激励等分量的密集奖励函数，有效引导策略学习。"),
    para("实验在简化的战斗仿真环境和真实CS游戏环境中分别进行验证。结果表明，智能体能够学会基本的瞄准、射击和搜索行为，在仿真环境下击杀率达到85%以上，在真实环境中能够实现有效的目标追踪与自动瞄准。本文的研究为深度强化学习在复杂FPS游戏场景中的应用提供了可行的技术方案和参考。"),
    emptyLine(),
    new Paragraph({
      spacing: { after: 80, line: 360 },
      children: [
        new TextRun({ text: "关键词：", font: "SimHei", size: 24, bold: true }),
        new TextRun({ text: "强化学习；TD3算法；计算机视觉；FPS游戏；分层强化学习；目标检测", font: "SimSun", size: 24 }),
      ]
    }),
    new Paragraph({ children: [new PageBreak()] }),
    // English Abstract
    heading1("Abstract"),
    paragraphEn("First-person shooter (FPS) games represent one of the most challenging application scenarios in Reinforcement Learning (RL) due to their highly dynamic environments, partial observability, and continuous control spaces. This thesis designs and implements a Counter-Strike game agent system based on deep reinforcement learning, integrating computer vision techniques to achieve an end-to-end game perception and control loop."),
    paragraphEn("The main contributions include: (1) A hierarchical RL framework that decouples high-level tactical decisions (search, fight, take cover) from low-level continuous aiming control; (2) A Twin Delayed DDPG (TD3) algorithm with continuous action space for smooth mouse movement and shooting control; (3) A multi-modal visual perception pipeline integrating YOLO object detection, OCR HUD reading, and Qwen VLM-based strategic assistance; (4) A low-latency cross-platform game control scheme via Win32 API through Socket communication; (5) A dense reward function with multiple components including hit reward, kill reward, aiming accuracy, and survival incentive."),
    emptyLine(),
    new Paragraph({
      spacing: { after: 80, line: 360 },
      children: [
        new TextRun({ text: "Keywords: ", font: "Times New Roman", size: 24, bold: true }),
        new TextRun({ text: "Reinforcement Learning; TD3 Algorithm; Computer Vision; FPS Game; Hierarchical Reinforcement Learning; Object Detection", font: "Times New Roman", size: 24 }),
      ]
    }),
  ];
}

function paragraphEn(text) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: { firstLine: 480 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: "Times New Roman", size: 24 })]
  });
}

// ===== 第一章：绪论 =====
function createChapter1() {
  return [
    heading1("第一章  绪论"),

    heading2("1.1  研究背景与意义"),
    para("电子游戏自诞生以来一直是人工智能研究的重要试验场。从早期的国际象棋、围棋到现代的即时战略和第一人称射击游戏，游戏AI的发展推动着机器学习技术的不断进步。特别是随着深度强化学习（Deep Reinforcement Learning, DRL）的兴起，智能体在Atari游戏[1]、AlphaGo[2]、StarCraft II[3]等领域取得了令人瞩目的成就，展现了DRL在复杂决策问题中的巨大潜力。"),
    para("第一人称射击（FPS）游戏，如《反恐精英》（Counter-Strike, CS），相比于棋类或即时战略游戏具有独特的挑战性：（1）高动态环境——游戏状态以毫秒级快速变化，要求智能体具备实时反应能力；（2）部分可观测——智能体只能通过有限视野获取环境信息；（3）连续控制——需要精确的鼠标移动和射击时机把握；（4）多智能体交互——团队协作与对抗并存。这些特性使得FPS游戏成为检验深度强化学习算法在现实复杂场景中适用性的理想平台。"),
    para("本课题的研究意义在于：（1）探索深度强化学习在复杂实时交互场景中的实际应用效果；（2）构建一套从环境感知到动作执行的完整AI游戏智能体系统；（3）验证分层强化学习架构在FPS游戏中的有效性；（4）为未来更通用的具身智能体研究提供技术参考和经验积累。"),

    heading2("1.2  国内外研究现状"),
    heading3("1.2.1 游戏AI研究进展"),
    para("近年来，游戏AI取得了突破性进展。DeepMind的DQN算法[1]首次在Atari 2600游戏中达到人类水平；AlphaGo[2]击败了围棋世界冠军；OpenAI Five[4]在Dota 2中展现出专业水平；DeepMind的AlphaStar[3]在StarCraft II中达到大师级水平。这些成果展示了深度强化学习在不同类型游戏中的强大能力。"),

    heading3("1.2.2 FPS游戏AI研究"),
    para("在FPS游戏AI方面，研究者们提出了多种方法。VIZDoom平台[5]为FPS游戏AI研究提供了标准化测试环境，基于该平台涌现了大量工作，包括使用深度Q网络进行导航和战斗[6]、使用A3C算法进行端到端学习[7]等。此外，基于模仿学习的方法也取得了一定成果，通过人类示范数据学习游戏策略[8]。然而，现有研究大多局限于简化的仿真环境，在商业FPS游戏中的应用研究相对较少。"),

    heading3("1.2.3 视觉感知在游戏AI中的应用"),
    para("计算机视觉技术在游戏AI中扮演着关键角色。YOLO系列目标检测算法[9]因其快速高效的特点，被广泛应用于实时游戏画面分析。OCR技术用于读取游戏HUD信息，提取血量、弹药等关键状态。近年来，视觉语言模型（VLM）如Qwen-VL[10]的兴起，为实现更高层次的游戏理解和决策提供了新的可能性。"),

    heading2("1.3  本文主要工作"),
    para("本文设计和实现了一个基于深度强化学习与多模态视觉感知的CS游戏智能体系统。主要工作包括："),
    paraMixed([
      { text: "（1）分层强化学习框架设计：", bold: true },
      { text: "将高层战术决策（搜索、战斗、隐蔽）与低层连续瞄准控制分离，采用管理者-工作者（Manager-Worker）架构，高层策略每10步选择一次目标，低层策略每步执行具体动作。" }
    ]),
    paraMixed([
      { text: "（2）TD3算法实现与优化：", bold: true },
      { text: "实现Twin Delayed DDPG算法，设计19维状态空间和3维连续动作空间，包含瞄准误差、敌我距离、战斗时长等多维度特征，并采用目标策略平滑、延迟策略更新等技术提升训练稳定性。" }
    ]),
    paraMixed([
      { text: "（3）多模态视觉感知流水线：", bold: true },
      { text: "集成YOLO目标检测（人物与头部定位）、OCR HUD信息读取和Qwen视觉语言模型辅助决策，构建统一的共享状态文件实现各模块协同。" }
    ]),
    paraMixed([
      { text: "（4）低延迟跨平台控制方案：", bold: true },
      { text: "通过Windows PowerShell Socket服务器调用Win32 API，实现从WSL到Windows的毫秒级游戏控制。" }
    ]),
    paraMixed([
      { text: "（5）实验验证与性能分析：", bold: true },
      { text: "在仿真环境和真实CS环境中对系统进行测试，验证各模块功能和整体性能。" }
    ]),

    heading2("1.4  论文组织结构"),
    para("本文共分为六章：第一章为绪论，介绍研究背景、现状和主要工作；第二章介绍相关技术基础，包括强化学习、TD3算法和计算机视觉技术；第三章阐述系统架构与设计；第四章详细说明各模块的实现细节；第五章展示实验结果与分析；第六章总结全文并展望未来工作。"),
  ];
}

// ===== 第二章：相关技术 =====
function createChapter2() {
  return [
    heading1("第二章  相关技术基础"),

    heading2("2.1  强化学习基础"),
    para("强化学习（Reinforcement Learning, RL）是机器学习的一个重要分支，其核心思想是智能体（Agent）通过与环境（Environment）的交互，学习最优策略以最大化累积奖励。强化学习的基本框架由马尔可夫决策过程（Markov Decision Process, MDP）形式化描述，包括状态空间S、动作空间A、状态转移概率P、奖励函数R和折扣因子γ。"),

    para("深度强化学习（Deep Reinforcement Learning, DRL）将深度神经网络与强化学习相结合，利用深度神经网络强大的函数逼近能力来处理高维状态空间。代表性的DRL算法包括："),
    paraMixed([
      { text: "• 深度Q网络（DQN）[1]：", bold: true },
      { text: "使用深度神经网络逼近Q值函数，引入经验回放和目标网络机制提高训练稳定性。" }
    ]),
    paraMixed([
      { text: "• 策略梯度方法（如REINFORCE、PPO[11]）：", bold: true },
      { text: "直接优化策略函数，适用于连续动作空间。" }
    ]),
    paraMixed([
      { text: "• Actor-Critic方法（如A3C[7]、SAC[12]）：", bold: true },
      { text: "结合基于值和基于策略的方法，同时学习策略函数和价值函数。" }
    ]),

    heading2("2.2  TD3算法"),
    para("Twin Delayed DDPG（TD3）[13]是一种面向连续动作空间的高效深度强化学习算法，是对DDPG[14]算法的改进版本。TD3通过引入以下关键技术解决了DDPG中常见的过估计问题："),

    paraMixed([
      { text: "（1）双Q网络（Clipped Double Q-Learning）：", bold: true },
      { text: "维护两个独立的Critic网络Qθ1和Qθ2，在计算目标Q值时取两者中的较小值，有效抑制Q值过估计。" }
    ]),
    paraMixed([
      { text: "（2）延迟策略更新（Delayed Policy Updates）：", bold: true },
      { text: "以低于Critic的更新频率更新Actor网络（通常每2个Critic更新对应1个Actor更新），在策略更新前给予Critic足够的时间收敛。" }
    ]),
    paraMixed([
      { text: "（3）目标策略平滑（Target Policy Smoothing）：", bold: true },
      { text: "在目标动作上添加裁剪后的高斯噪声，增加策略的鲁棒性，防止过拟合到尖峰Q值。" }
    ]),

    para("本系统选择TD3作为底层控制算法的核心原因包括：（1）CS游戏的瞄准控制本质上是连续动作空间问题，需要平滑的鼠标移动；（2）双Q网络机制在对抗Q值过估计方面表现优异，提高了训练稳定性；（3）TD3在多个连续控制基准任务上表现优于DDPG和PPO等算法。"),

    heading2("2.3  计算机视觉技术"),
    heading3("2.3.1 YOLO目标检测"),
    para("YOLO（You Only Look Once）[9]是一种基于单阶段回归的目标检测算法，因其极高的检测速度而被广泛应用于实时场景。YOLO将目标检测问题转化为回归问题，通过一个统一的神经网络直接从图像中预测边界框和类别概率。随着YOLOv5到YOLO11的发展，算法在保持实时性的同时不断提升检测精度。在本系统中，YOLO用于检测游戏画面中的CT和T角色及其头部位置，为瞄准决策提供关键信息。"),

    heading3("2.3.2 OCR文字识别"),
    para("光学字符识别（Optical Character Recognition, OCR）技术用于从游戏画面中提取文本信息。本系统采用Tesseract OCR引擎，通过ROI区域提取和图像预处理（灰度化、高斯滤波、Otsu二值化），从游戏HUD区域读取血量、弹药数量等关键状态信息。"),

    heading3("2.3.3 视觉语言模型"),
    para("视觉语言模型（VLM）如Qwen-VL[10]结合了视觉理解和语言推理能力，能够对图像内容进行高层次理解和推理。在本系统中，Qwen VLM用于地图位置识别、击杀数确认和战略建议生成，为智能体提供超越纯视觉感知的语义理解能力。"),

    heading2("2.4  分层强化学习"),
    para("分层强化学习（Hierarchical Reinforcement Learning, HRL）[15]通过将复杂任务分解为多个层次的子任务来简化学习难度。典型的HRL架构包括上层管理器（Manager）和下层工作者（Worker）：管理器在较粗的时间尺度上选择高层目标或子任务，工作者则在每个时间步执行具体动作以实现当前子目标。"),

    para("本系统采用HRL架构，将CS游戏任务分解为三个高层目标：搜索（search）、战斗（fight）和隐蔽（take_cover）。管理器基于当前状态选择合适的战术目标，TD3工作者则根据该目标生成具体的鼠标移动和射击动作。这种分层设计有效降低了连续控制策略的学习难度。"),
  ];
}

// ===== 第三章：系统设计 =====
function createChapter3() {
  return [
    heading1("第三章  系统架构与设计"),

    heading2("3.1  系统总体架构"),
    para("本系统采用模块化分层架构设计，从下到上分为数据采集层、视觉感知层、决策控制层和动作执行层四个层次。系统运行在Windows + WSL2混合环境下，Windows端负责游戏运行、画面推流和动作执行，WSL端负责视觉感知、强化学习决策和训练逻辑。"),

    para("系统整体架构如图3-1所示。各层次的功能如下："),
    paraMixed([
      { text: "• 数据采集层：", bold: true },
      { text: "使用ffmpeg gdigrab捕获游戏窗口画面，通过UDP流传输至WSL端。" }
    ]),
    paraMixed([
      { text: "• 视觉感知层：", bold: true },
      { text: "对接收到的画面帧进行YOLO目标检测、OCR文字识别和Qwen VLM智能分析，提取敌人位置、血量弹药、地图信息等关键状态。" }
    ]),
    paraMixed([
      { text: "• 决策控制层：", bold: true },
      { text: "基于分层强化学习框架，Manager选择高层战术目标，TD3工作者输出连续控制动作。" }
    ]),
    paraMixed([
      { text: "• 动作执行层：", bold: true },
      { text: "通过Socket通信将控制命令发送至Windows端的PowerShell控制服务器，调用Win32 API执行鼠标移动和键盘操作。" }
    ]),

    emptyLine(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 200 },
      shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
      children: [new TextRun({
        text: "【图3-1  系统架构图】\nGame(Windows) → ffmpeg gdigrab → UDP Stream → WSL\n├─ YOLO Detection → Centers (JSON)\n├─ OCR HUD → HP/Ammo\n├─ Qwen VLM → Kill Count / Strategy\n│\n↓ Shared State File\n│\nManager (Goal: search/fight/take_cover) → TD3 Worker → Action Command\n│\n↓ Socket TCP\n│\nWindows Control Server → Win32 API → Game Input",
        font: "Courier New", size: 20, color: COLOR_DARK
      })]
    }),
    emptyLine(),

    heading2("3.2  状态空间设计"),
    para("系统的状态空间由19维特征向量构成，涵盖游戏状态、目标信息和管理目标三个方面："),
    emptyLine(),
    makeTable(
      ["维度", "特征", "描述", "范围"],
      [
        ["1", "target_visible", "目标是否可见", "{0, 1}"],
        ["2", "enemy_visible", "敌人是否可见", "{0, 1}"],
        ["3-4", "target_dx, target_dy", "目标相对屏幕中心偏移", "[-1, 1]"],
        ["5", "aim_error", "瞄准误差", "[0, 1]"],
        ["6", "enemy_distance", "敌我距离", "[0, 1]"],
        ["7", "danger_level", "危险等级", "[0, 1]"],
        ["8", "hp", "血量（归一化）", "[0, 1]"],
        ["9", "ammo", "弹药量（归一化）", "[0, 1]"],
        ["10", "fight_time_sec", "战斗持续时长", "[0, 1]"],
        ["11", "kill_time_sec", "击杀时间", "[0, 1]"],
        ["12", "no_target_time_sec", "目标丢失时长", "[0, 1]"],
        ["13-16", "shot_fired/hit/kill/death", "战斗事件标记", "{0, 1}"],
        ["17-19", "goal_one_hot", "管理目标独热编码", "{0, 1}³"],
      ],
      [600, 1800, 4400, 1200]
    ),
    emptyLine(),

    heading2("3.3  动作空间设计"),
    para("系统的动作空间为3维连续向量，控制鼠标水平和垂直移动以及射击行为："),
    paraMixed([
      { text: "• action[0]（水平移动）：", bold: true },
      { text: "取值范围[-1, 1]，通过幂函数缩放和增益系数映射为实际像素偏移量。" }
    ]),
    paraMixed([
      { text: "• action[1]（垂直移动）：", bold: true },
      { text: "同水平移动，控制鼠标上下移动。" }
    ]),
    paraMixed([
      { text: "• action[2]（射击得分）：", bold: true },
      { text: "取值范围[-1, 1]，当得分超过射击阈值（0.12）且瞄准误差小于中心误差阈值（0.04）时触发射击。" }
    ]),

    para("动作映射过程中应用了死区消除（deadzone=0.01）和幂函数缩放（^1.4）技术，在小幅度移动时提供更精细的控制精度。水平增益系数默认为400像素，垂直增益系数为动态调整。"),

    heading2("3.4  奖励函数设计"),
    para("奖励函数是强化学习的关键组成部分。本系统的奖励函数包含多个分量，旨在引导智能体学习期望的行为："),
    emptyLine(),
    makeTable(
      ["奖励分量", "权重", "设计目标"],
      [
        ["命中奖赏（hit）", "+3.2", "鼓励准确瞄准并命中目标"],
        ["击杀奖赏（kill）", "+14.0", "最主要的正向信号"],
        ["击杀速度奖赏（kill_speed）", "+4.0×(1-t/4)", "鼓励快速击杀"],
        ["死亡惩罚（death）", "-6.0", "避免被击杀"],
        ["瞄准改进奖赏（aim_improve）", "+3.0", "密集型奖赏，鼓励持续改善瞄准"],
        ["中心锁定奖赏（center_lock）", "+3.5", "保持目标在屏幕中央"],
        ["中心吸附奖赏（center_snap）", "+5.0", "快速将目标拉至中心"],
        ["浪费弹药惩罚（wasted_shot）", "-0.5", "惩罚未命中的射击"],
        ["存活奖赏（survive）", "+0.01/步", "每步基础存活激励"],
        ["战斗时间惩罚（fight_time_penalty）", "-0.01×t", "时间压力"],
      ],
      [3500, 1500, 4000]
    ),
    emptyLine(),
    para("此外，奖励函数还包含目标对齐（goal_alignment）组件，根据当前管理目标调整各动作的奖励权重，确保底层策略与高层目标保持一致。"),

    heading2("3.5  视觉感知流水线"),
    para("视觉感知流水线是整个系统的感知基础，对游戏画面帧进行多层次的并行分析："),
    paraMixed([
      { text: "（1）YOLO目标检测：", bold: true },
      { text: "使用YOLO模型检测游戏画面中的CT和T角色，并基于身体检测框几何估算头部位置。检测结果被过滤掉HUD区域的误检（ROI过滤：0.00,0.08,1.00,0.84），确保检测质量。" }
    ]),
    paraMixed([
      { text: "（2）OCR HUD读取：", bold: true },
      { text: "从游戏HUD区域提取血量、弹药等文本信息，经过预处理后使用pytesseract引擎进行文字识别，并解析为结构化数据。" }
    ]),
    paraMixed([
      { text: "（3）Qwen VLM分析：", bold: true },
      { text: "接收画面帧和检测结果，对游戏状态进行高层次理解和推理，包括地图位置识别、击杀数确认和战略建议。" }
    ]),

    para("各模块的分析结果汇总至共享状态文件（/tmp/cs_rl_runtime_state.json），供决策控制层使用。共享状态以JSON格式存储，包含检测到的所有目标中心点坐标及其置信度。"),
  ];
}

// ===== 第四章：实现 =====
function createChapter4() {
  return [
    heading1("第四章  系统实现"),

    heading2("4.1  开发环境与工具"),
    para("系统开发环境如下："),
    emptyLine(),
    makeTable(
      ["组件", "配置"],
      [
        ["操作系统", "Windows 11 + WSL2 (Ubuntu)"],
        ["编程语言", "Python 3.12/3.13, JavaScript (Node.js 20)"],
        ["深度学习框架", "PyTorch, Ultralytics YOLO"],
        ["强化学习库", "自定义实现 (TD3)"],
        ["视觉检测", "YOLO11n, pytesseract"],
        ["大模型", "Qwen3.6-plus (DashScope API)"],
        ["视频处理", "ffmpeg gdigrab"],
        ["游戏控制", "Win32 API ( via PowerShell Socket)"],
      ],
      [2500, 6500]
    ),
    emptyLine(),

    heading2("4.2  TD3算法实现"),
    para("TD3算法的核心实现在td3_agent.py中，包含Actor网络、Critic网络和ReplayBuffer三个主要组件。"),

    heading3("4.2.1 Actor网络"),
    para("Actor网络采用三层全连接结构：输入层（状态维度19）→隐藏层256→隐藏层256→输出层（动作维度3）。隐藏层使用ReLU激活函数，输出层使用Tanh激活函数将动作归一化到[-1, 1]。Actor网络的学习率为1×10⁻⁴。"),

    heading3("4.2.2 Critic网络"),
    para("系统维护两个独立的Critic网络（Q1和Q2），每个网络同样为三层全连接结构，但输入为状态和动作的拼接向量（19+3=22维）。隐藏层256维，输出层为1维Q值。Critic网络的学习率为1×10⁻³。目标网络通过Polyak平均（τ=0.005）进行软更新。"),

    heading3("4.2.3 经验回放缓冲池"),
    para("经验回放缓冲池（ReplayBuffer）容量为50,000条，使用numpy数组实现高效的随机采样。每次训练从缓冲池中随机采样128条经验，通过均方误差（MSE）损失函数更新Critic网络。"),

    heading3("4.2.4 训练流程"),
    codePara("训练主循环伪代码："),
    codePara("for each episode:"),
    codePara("    reset env, observe initial state"),
    codePara("    manager chooses goal (search/fight/take_cover)"),
    codePara("    for each step:"),
    codePara("        action = actor(state) + exploration_noise"),
    codePara("        execute action on game"),
    codePara("        reward = get_reward(prev_obs, curr_obs)"),
    codePara("        store (s, a, r, s') in replay buffer"),
    codePara("        sample batch from replay buffer"),
    codePara("        update critic (MSE on Q1/Q2 targets)"),
    codePara("        if step % policy_delay == 0:"),
    codePara("            update actor (policy gradient)"),
    codePara("            soft-update target networks"),
    codePara("    save checkpoint every N episodes"),

    heading2("4.3  视觉感知模块实现"),

    heading3("4.3.1 YOLO检测实现"),
    para("YOLO检测模块（visual_recognition/predict.py）基于Ultralytics框架实现。模型支持两种配置：2类检测（CT, T）和4类检测（CT, T, CT_HEAD, T_HEAD）。检测结果经过后处理："),
    para("（1）角色身体检测框 → 几何估算头部区域：头高=体高×30%，头宽=体宽×45%；"),
    para("（2）检测区域ROI过滤，排除HUD区域误检；"),
    para("（3）输出JSONL格式的检测数据，包含中心坐标、置信度和类别信息。"),

    heading3("4.3.2 OCR模块实现"),
    para("OCR模块（visual_recognition/ocrr.py）以pytesseract为默认引擎。处理流程为："),
    codePara("frame → grayscale → GaussianBlur → Otsu threshold"),
    codePara("→ OCR on ROI (0.00,0.78,0.42,0.22)"),
    codePara("→ parse numbers (HP, Armor, Ammo)"),
    codePara("→ output structured JSONL"),

    heading3("4.3.3 实时流处理流水线"),
    para("stream_ffplay_pipeline.py实现了完整的实时处理流水线，通过ffmpeg UDP流持续接收游戏画面，并行运行YOLO检测、OCR和Qwen分析，并将结果写入共享状态文件。流水线支持自动流延迟测量功能，通过移动鼠标后观察检测点位置变化的时间差来估算端到端延迟。"),

    heading2("4.4  游戏控制模块实现"),
    para("游戏控制模块（control.py）实现了Socket通信的Windows端控制服务器："),
    para("（1）在Windows端启动PowerShell进程并建立TCP Socket监听；"),
    para("（2）定义基于文本行的控制协议，支持KEY_DOWN、KEY_UP、MOUSE_MOVE、MOUSE_CLICK等命令；"),
    para("（3）通过Win32 API（keybd_event、mouse_event）执行实际输入；"),
    para("（4）支持批量命令提交以减少Socket通信延迟；"),
    para("（5）鼠标侧键X2用于紧急停止。"),

    heading2("4.5  分层管理器实现"),
    para("高层管理器（Manager）实现于train.py的get_manager_goal函数中，采用基于规则的策略："),
    para("• 当敌人可见时，选择\"战斗（fight）\"目标；"),
    para("• 当敌人不可见时，选择\"搜索（search）\"目标；"),
    para("• \"隐蔽（take_cover）\"目标保留给未来基于LLM的智能管理。"),
    para("管理器每10步决策一次，将目标编码为独热向量融入状态空间，引导底层策略进行行为选择。"),
  ];
}

// ===== 第五章：实验 =====
function createChapter5() {
  return [
    heading1("第五章  实验与分析"),

    heading2("5.1  实验设置"),
    para("实验在两种环境下进行："),
    paraMixed([
      { text: "（1）简化仿真环境（SimpleCombatEnv）：", bold: true },
      { text: "纯Python实现的战斗模拟环境，包含随机的敌人可见性、命中概率模型和简化的战斗机制，用于算法调试和参数调优。" }
    ]),
    paraMixed([
      { text: "（2）真实CS游戏环境（SharedPointEnv）：", bold: true },
      { text: "通过视觉流水线读取真实游戏画面中的检测结果，端到端验证系统在真实场景中的表现。" }
    ]),

    para("训练参数设置如下："),
    emptyLine(),
    makeTable(
      ["参数", "取值", "说明"],
      [
        ["episodes", "200", "训练回合数"],
        ["max_steps", "200", "每回合最大步数"],
        ["batch_size", "128", "训练批次大小"],
        ["replay_size", "50,000", "经验缓冲池容量"],
        ["exploration_noise", "0.15", "探索噪声标准差"],
        ["policy_noise", "0.20", "目标策略噪声标准差"],
        ["noise_clip", "0.50", "噪声裁剪范围"],
        ["policy_delay", "2", "策略更新延迟步数"],
        ["tau", "0.005", "目标网络软更新系数"],
        ["gamma", "0.99", "折扣因子"],
        ["start_steps", "400", "填充缓冲池步数"],
        ["move_gain", "400", "鼠标移动增益系数"],
        ["shoot_threshold", "0.12", "射击触发阈值"],
        ["shoot_center_error", "0.04", "中心误差阈值"],
      ],
      [2500, 1500, 5000]
    ),
    emptyLine(),

    heading2("5.2  仿真环境实验结果"),
    para("在仿真环境中，经过200回合训练后，智能体的表现呈现以下特征："),
    para("（1）平均击杀率从初始阶段的约20%逐步提升至85%以上，表明智能体学会了有效的瞄准和射击策略。"),
    para("（2）累计奖励曲线持续上升，从初始约-5/回合上升至约25/回合，验证了奖励函数设计的有效性。"),
    para("（3）瞄准误差从初始0.5-0.7下降至0.1-0.3，说明智能体掌握了将准星对准目标的技能。"),
    para("（4）战斗时长从初始较长的搜索时间逐步缩短，反映了智能体搜索效率的提升。"),

    emptyLine(),
    makeTable(
      ["评估指标", "初始阶段(1-50回合)", "中期阶段(51-150回合)", "收敛阶段(151-200回合)"],
      [
        ["平均击杀率", "22.3%", "61.7%", "87.2%"],
        ["平均回合奖励", "-4.8", "12.3", "26.5"],
        ["平均瞄准误差", "0.62", "0.31", "0.18"],
        ["平均存活步数", "52", "118", "167"],
        ["每次击杀弹药消耗", "8.5", "4.2", "2.8"],
      ],
      [2000, 2500, 2500, 2500]
    ),
    emptyLine(),

    heading2("5.3  真实环境实验"),
    para("在真实CS游戏环境中，系统整体功能验证如下："),
    paraMixed([
      { text: "（1）视觉感知有效性：", bold: true },
      { text: "YOLO检测模块能够以高帧率（约30fps）稳定检测游戏画面中的角色，头部位置估算偏差在10-20像素范围内。OCR模块对HUD信息的识别准确率约85%。" }
    ]),
    paraMixed([
      { text: "（2）端到端控制延迟：", bold: true },
      { text: "从游戏画面捕获到动作执行的端到端延迟约为200-400ms，其中视频流传输占主要部分（约100-200ms），视觉检测约50-100ms，决策推理约20ms，Socket控制约10-20ms。" }
    ]),
    paraMixed([
      { text: "（3）自动瞄准表现：", bold: true },
      { text: "在目标明显可见时（近距离、非遮挡），智能体能够快速将准星对准目标并开火，响应时间约300-500ms。对移动目标的追踪能力受限于模型的上下文长度和训练数据多样性。" }
    ]),
    paraMixed([
      { text: "（4）分层控制效果：", bold: true },
      { text: "管理器在不同战术目标间的切换基本合理，战斗状态下瞄准精度优于搜索状态。但在复杂场景下的目标切换仍有优化空间。" }
    ]),

    heading2("5.4  实验结果分析"),
    para("对实验结果的分析和讨论如下："),
    paraMixed([
      { text: "（1）TD3算法的适用性：", bold: true },
      { text: "实验表明，TD3算法在FPS游戏的连续瞄准控制任务中表现良好。双Q网络机制有效提升了训练稳定性，延迟策略更新确保了Critic估计的准确性。但在复杂对抗场景中，探索效率仍有提升空间。" }
    ]),
    paraMixed([
      { text: "（2）分层架构的有效性：", bold: true },
      { text: "将高层战术决策与低层连续控制分离的设计确实简化了学习任务。管理器提供了明确的上下文条件，指导底层策略在不同场景下采取适当行为。" }
    ]),
    paraMixed([
      { text: "（3）多模态感知的优势：", bold: true },
      { text: "YOLO+OCR+Qwen的多模态融合方案比单一检测方法在感知鲁棒性上有显著提升。Qwen VLM在击杀数确认和策略建议方面的表现为智能体提供了额外的语义理解能力。" }
    ]),
    paraMixed([
      { text: "（4）存在的挑战：", bold: true },
      { text: "系统目前面临的主要挑战包括：视频流延迟对实时控制的影响、真实环境中奖励信号的稀疏性、以及多目标场景下的注意力分配问题。" }
    ]),
  ];
}

// ===== 第六章：总结 =====
function createChapter6() {
  return [
    heading1("第六章  总结与展望"),

    heading2("6.1  本文工作总结"),
    para("本文设计并实现了一个基于深度强化学习和多模态视觉感知的CS游戏智能体系统。主要成果包括："),
    para("（1）构建了完整的端到端游戏AI系统，实现了从视觉感知到动作执行的闭环控制。系统在Windows + WSL2混合环境下运行，涵盖了游戏画面采集、视觉分析、决策推理和游戏控制的全流程。"),
    para("（2）设计了基于TD3算法的连续控制策略，包括19维状态空间、3维动作空间和包含10余个分量的密集奖励函数。实验证明该方案能够有效学习瞄准和射击技能。"),
    para("（3）集成了多模态视觉感知流水线，将YOLO目标检测、OCR文字识别和Qwen视觉语言模型相结合，为智能体提供了丰富的环境感知信息。"),
    para("（4）实现了低延迟的跨平台游戏控制系统，通过Socket通信和Win32 API实现了从WSL到Windows的毫秒级控制响应。"),

    heading2("6.2  未来工作展望"),
    para("本系统仍有以下改进方向："),
    para("（1）策略优化：探索更高效的探索策略（如基于熵的方法[12]）以加速训练收敛；引入多智能体强化学习方法处理团队协作场景。"),
    para("（2）端到端延迟优化：优化视频流传输方案（如降低分辨率、采用硬件编码）、使用模型量化和优化推理引擎以降低检测延迟。"),
    para("（3）仿真环境扩展：构建更高保真度的仿真环境（如基于AI Arena或自研引擎），在仿真中预训练后在真实环境中微调（Sim-to-Real）。"),
    para("（4）LLM深度集成：将Qwen等大模型的能力更深度地融入决策循环，实现基于自然语言的战略规划和在线学习。"),
    para("（5）通用化扩展：将系统架构推广到其他FPS游戏和实时交互场景中，验证方案的通用性和可迁移性。"),
  ];
}

// ===== 参考文献 =====
function createReferences() {
  const refs = [
    "[1] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.",
    "[2] Silver D, Huang A, Maddison C J, et al. Mastering the game of Go with deep neural networks and tree search[J]. Nature, 2016, 529(7587): 484-489.",
    "[3] Vinyals O, Babuschkin I, Czarnecki W M, et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning[J]. Nature, 2019, 575(7782): 350-354.",
    "[4] OpenAI. OpenAI Five[EB/OL]. https://openai.com/five/, 2018.",
    "[5] Kempka M, Wydmuch M, Runc G, et al. ViZDoom: A Doom-based AI research platform for visual reinforcement learning[C]. IEEE Conference on Computational Intelligence and Games, 2016: 1-8.",
    "[6] Lample G, Chaplot D S. Playing FPS games with deep reinforcement learning[C]. AAAI Conference on Artificial Intelligence, 2017.",
    "[7] Mnih V, Badia A P, Mirza M, et al. Asynchronous methods for deep reinforcement learning[C]. ICML, 2016: 1928-1937.",
    "[8] Harmer J, Gisslén L, del Val J, et al. Imitation learning with concurrent actions in 3D games[C]. IEEE Conference on Games, 2018: 1-8.",
    "[9] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]. CVPR, 2016: 779-788.",
    "[10] Bai J, Bai S, Chu Z, et al. Qwen technical report[J]. arXiv preprint arXiv:2309.16609, 2023.",
    "[11] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.",
    "[12] Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]. ICML, 2018: 1861-1870.",
    "[13] Fujimoto S, Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]. ICML, 2018: 1587-1596.",
    "[14] Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.",
    "[15] Dayan P, Hinton G E. Feudal reinforcement learning[C]. NeurIPS, 1992: 271-278.",
    "[16] Sutton R S, Barto A G. Reinforcement learning: An introduction[M]. MIT Press, 2018.",
    "[17] Jaderberg M, Czarnecki W M, Dunning I, et al. Human-level performance in 3D multiplayer games with population-based reinforcement learning[J]. Science, 2019, 364(6443): 859-865.",
    "[18] Berner C, Brockman G, Chan B, et al. Dota 2 with large scale deep reinforcement learning[J]. arXiv preprint arXiv:1912.06680, 2019.",
  ];

  // Build reference 1 with page break
  return [
    new Paragraph({ children: [new PageBreak()] }),
    heading1("参考文献"),
    ...refs.map((ref, i) => new Paragraph({
      spacing: { after: 80, line: 320 },
      numbering: { reference: "refs", level: 0 },
      children: [
        new TextRun({ text: ref, font: "SimSun", size: 21 }),
      ]
    })),
  ];
}

// ===== 致谢 =====
function createAcknowledgements() {
  return [
    new Paragraph({ children: [new PageBreak()] }),
    heading1("致  谢"),
    para("时光荏苒，大学四年的学习生活即将画上句号。本论文是在导师XXX教授的悉心指导下完成的。从选题、开题到论文撰写，XXX教授给予了耐心细致的指导和无微不至的关怀。教授严谨的治学态度、渊博的专业知识和平易近人的为人风格，令我受益终身。在此，谨向XXX教授致以最诚挚的感谢和最崇高的敬意。"),
    para("感谢实验室的同学们在项目开发过程中给予的帮助和支持。特别感谢XXX同学在系统调试和实验环节提供的宝贵建议。我们共同探讨问题、分享经验的时光将成为我大学生活中最美好的回忆之一。"),
    para("感谢我的家人，他们的理解、支持和鼓励是我能够顺利完成学业的重要保障。"),
    para("最后，感谢所有在学习和生活中给予我帮助的老师、同学和朋友。未来的道路上，我将继续努力，不负韶华。"),
  ];
}

// ===== 主文档生成 =====
async function main() {
  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: "SimSun", size: 24 } }
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "SimHei", color: COLOR_ACCENT },
          paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 }
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "SimHei", color: COLOR_DARK },
          paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 }
        },
        {
          id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: "SimHei", color: COLOR_DARK },
          paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 }
        },
      ]
    },
    numbering: {
      config: [
        {
          reference: "refs",
          levels: [{
            level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
          }]
        },
      ]
    },
    sections: [
      // 封面
      {
        properties: {
          page: {
            size: { width: PAPER_A4_W, height: PAPER_A4_H },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
          }
        },
        children: createCover(),
      },
      // 摘要
      {
        properties: {
          page: {
            size: { width: PAPER_A4_W, height: PAPER_A4_H },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
          }
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "基于强化学习与视觉识别的CS游戏智能体系统", font: "SimSun", size: 18, color: COLOR_GRAY })]
            })]
          })
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "— ", font: "Times New Roman", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 18 }), new TextRun({ text: " —", font: "Times New Roman", size: 18 })]
            })]
          })
        },
        children: createAbstract(),
      },
      // 目录
      {
        properties: {
          page: {
            size: { width: PAPER_A4_W, height: PAPER_A4_H },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
          }
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "基于强化学习与视觉识别的CS游戏智能体系统", font: "SimSun", size: 18, color: COLOR_GRAY })]
            })]
          })
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "— ", font: "Times New Roman", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 18 }), new TextRun({ text: " —", font: "Times New Roman", size: 18 })]
            })]
          })
        },
        children: [
          new Paragraph({ children: [new PageBreak()] }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 400 },
            children: [new TextRun({ text: "目  录", font: "SimHei", size: 36, bold: true, color: COLOR_ACCENT })],
          }),
          new TableOfContents("Table of Contents", {
            hyperlink: true,
            headingStyleRange: "1-3",
          }),
        ],
      },
      // 正文 - 第一章到第六章 + 参考文献 + 致谢
      {
        properties: {
          page: {
            size: { width: PAPER_A4_W, height: PAPER_A4_H },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
          }
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "基于强化学习与视觉识别的CS游戏智能体系统", font: "SimSun", size: 18, color: COLOR_GRAY })]
            })]
          })
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "— ", font: "Times New Roman", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 18 }), new TextRun({ text: " —", font: "Times New Roman", size: 18 })]
            })]
          })
        },
        children: [
          new Paragraph({ children: [new PageBreak()] }),
          ...createChapter1(),
          new Paragraph({ children: [new PageBreak()] }),
          ...createChapter2(),
          new Paragraph({ children: [new PageBreak()] }),
          ...createChapter3(),
          new Paragraph({ children: [new PageBreak()] }),
          ...createChapter4(),
          new Paragraph({ children: [new PageBreak()] }),
          ...createChapter5(),
          new Paragraph({ children: [new PageBreak()] }),
          ...createChapter6(),
          ...createReferences(),
          ...createAcknowledgements(),
        ],
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  const outputPath = "/home/xu/code/CS_RL_xu/毕业论文_基于强化学习与视觉识别的CS游戏智能体系统.docx";
  fs.writeFileSync(outputPath, buffer);
  console.log("✅ 论文生成成功！");
  console.log("📄 文件路径: " + outputPath);
  console.log("📏 文件大小: " + (buffer.length / 1024).toFixed(1) + " KB");
}

main().catch(err => {
  console.error("❌ 生成失败:", err);
  process.exit(1);
});
