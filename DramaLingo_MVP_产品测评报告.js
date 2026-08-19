const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TableOfContents
} = require("docx");

// ── Color Palette ──
const C = {
  brand: "7B2FBE",
  brand2: "C850C0",
  gold: "E8A020",
  green: "16A34A",
  red: "DC2626",
  blue: "2563EB",
  orange: "D97706",
  bg: "F8F5FF",
  muted: "8B7AAD",
  text: "1A1A2E",
  lightPurple: "EDE9F8",
  lightGreen: "DCFCE7",
  lightRed: "FEE2E2",
  lightOrange: "FEF3C7",
  lightBlue: "DBEAFE",
  white: "FFFFFF",
};

const border = { style: BorderStyle.SINGLE, size: 1, color: C.lightPurple };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: C.white };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

// Page: US Letter
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9360

// ── Helpers ──
function heading(text, level) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160 },
    ...opts,
    children: [new TextRun({ size: 22, font: "Arial", ...opts.run, text })],
  });
}
function bold(text, opts = {}) {
  return new TextRun({ bold: true, size: 22, font: "Arial", ...opts, text });
}
function run(text, opts = {}) {
  return new TextRun({ size: 22, font: "Arial", ...opts, text });
}
function bullet(text, ref = "bullets", level = 0) {
  return new Paragraph({
    numbering: { reference: ref, level },
    spacing: { after: 80 },
    children: [run(text)],
  });
}
function spacer(h = 200) {
  return new Paragraph({ spacing: { after: h }, children: [] });
}

// ── Score badge table cell ──
function scoreCell(score, label, color) {
  return new TableCell({
    borders: noBorders,
    width: { size: Math.floor(CONTENT_W / 5), type: WidthType.DXA },
    shading: { fill: color, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 80, right: 80 },
    verticalAlign: "center",
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: score, bold: true, size: 36, font: "Arial", color: C.text })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: label, size: 18, font: "Arial", color: C.muted })],
      }),
    ],
  });
}

// ── Issue Row ──
function issueRow(id, severity, module, title, description) {
  const sevColor = severity === "P0" ? C.lightRed : severity === "P1" ? C.lightOrange : C.lightBlue;
  const sevTextColor = severity === "P0" ? C.red : severity === "P1" ? C.orange : C.blue;
  const colWidths = [500, 600, 1200, 2200, 4860];
  return new TableRow({
    children: [
      new TableCell({
        borders, margins: cellMargins,
        width: { size: colWidths[0], type: WidthType.DXA },
        children: [new Paragraph({ children: [run(id, { size: 20, color: C.muted })] })],
      }),
      new TableCell({
        borders, margins: cellMargins,
        width: { size: colWidths[1], type: WidthType.DXA },
        shading: { fill: sevColor, type: ShadingType.CLEAR },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run(severity, { size: 20, bold: true, color: sevTextColor })] })],
      }),
      new TableCell({
        borders, margins: cellMargins,
        width: { size: colWidths[2], type: WidthType.DXA },
        children: [new Paragraph({ children: [run(module, { size: 20 })] })],
      }),
      new TableCell({
        borders, margins: cellMargins,
        width: { size: colWidths[3], type: WidthType.DXA },
        children: [new Paragraph({ children: [bold(title, { size: 20 })] })],
      }),
      new TableCell({
        borders, margins: cellMargins,
        width: { size: colWidths[4], type: WidthType.DXA },
        children: [new Paragraph({ children: [run(description, { size: 20 })] })],
      }),
    ],
  });
}

// ══════════════════════════════════════════════
// DOCUMENT
// ══════════════════════════════════════════════
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22, color: C.text } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: C.brand },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: C.brand },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: C.text },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "bullets2",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "○", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 360 } } },
        }],
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  sections: [
    // ════════════════ COVER PAGE ════════════════
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      children: [
        spacer(1200),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
          children: [new TextRun({ text: "DramaLingo MVP", size: 56, bold: true, font: "Arial", color: C.brand })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "产品深度测评报告", size: 40, bold: true, font: "Arial", color: C.text })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
          children: [run("用影视台词学地道英语", { size: 24, color: C.muted, italics: true })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: C.lightPurple } },
          children: [],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [run("测评角色：资深产品经理 + 英语学习爱好者", { size: 22, color: C.muted })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [run("测评日期：2026-05-23", { size: 22, color: C.muted })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [run("测评范围：用户检索页 + 管理后台全模块", { size: 22, color: C.muted })],
        }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },

    // ════════════════ BODY ════════════════
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: C.lightPurple } },
            children: [
              run("DramaLingo MVP · 产品测评报告", { size: 18, color: C.muted }),
            ],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 1, color: C.lightPurple } },
            children: [
              run("第 ", { size: 18, color: C.muted }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.muted, font: "Arial" }),
              run(" 页", { size: 18, color: C.muted }),
            ],
          })],
        }),
      },
      children: [
        // ── TOC ──
        new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════
        // 1. EXECUTIVE SUMMARY
        // ════════════════════════════════════════════
        heading("一、测评总览", HeadingLevel.HEADING_1),
        para("DramaLingo MVP 是一款基于影视台词的英语学习工具。用户输入中文描述或英文关键词，系统通过多信号语义检索返回最地道的影视台词，配合视频切片提供“听+看+读”的沉浸式学习体验。本报告从“用户体验友好性”“学习便捷性”“学习有效性”三个维度对用户端与管理后台进行全面测评。"),

        // Score cards
        new Table({
          width: { size: CONTENT_W, type: WidthType.DXA },
          columnWidths: Array(5).fill(Math.floor(CONTENT_W / 5)),
          rows: [new TableRow({
            children: [
              scoreCell("B+", "用户体验", C.lightGreen),
              scoreCell("B", "学习便捷性", C.lightBlue),
              scoreCell("B-", "学习有效性", C.lightOrange),
              scoreCell("A-", "后台功能", C.bg),
              scoreCell("B", "综合评分", C.lightPurple),
            ],
          })],
        }),
        spacer(200),
        para("产品核心价值主张“用影视台词学地道英语”已初步成型，搜索体验流畅、AI 思考过程可视化做得出色，但在学习闭环、用户留存设计、移动端适配等方面仍有显著提升空间。以下是详细的逐模块分析。"),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════
        // 2. USER PAGE EVALUATION
        // ════════════════════════════════════════════
        heading("二、用户检索页测评", HeadingLevel.HEADING_1),

        // 2.1 Strengths
        heading("2.1 亮点与优势", HeadingLevel.HEADING_2),

        heading("2.1.1 AI 思考过程可视化（⭐ 特色功能）", HeadingLevel.HEADING_3),
        para("搜索时展示“理解意图 → 语义补全 → 候选翻译 → 向量检索”四步动画，让用户感知到 AI 正在“思考”而非简单关键词匹配，显著提升信任感。搜索完成后“思考摘要卡”展示语义补全、候选英文、意图标签等内容，可折叠、不干扰主结果，设计很精细。"),

        heading("2.1.2 多维筛选系统", HeadingLevel.HEADING_3),
        para("提供难度、场景、是否有视频切片、每页数量四维筛选，布局紧凑、交互友好。切换筛选自动触发搜索，无需额外点击。"),

        heading("2.1.3 示例快捷入口", HeadingLevel.HEADING_3),
        para("“试试：表达感激 / 道歉 / 鼓励朋友 / 拒绝邀请 / 表白心意 / I love you / hope / friendship” 覆盖了常见学习场景，降低新用户上手门槛。支持中文和英文混合输入。"),

        heading("2.1.4 结果卡片信息密度合理", HeadingLevel.HEADING_3),
        para("每张卡片包含英文台词、中文翻译、相似度分数、剧集/难度/场景徽章、视频切片播放器，信息密度大但不繁琐。"),

        // 2.2 Issues
        heading("2.2 问题与优化建议", HeadingLevel.HEADING_2),

        heading("2.2.1 用户体验友好性", HeadingLevel.HEADING_3),

        // U1: Fake pagination
        new Paragraph({
          spacing: { before: 160, after: 80 },
          children: [bold("U1 分页失效（伪分页）", { color: C.red, size: 22 }), run(" —— P0", { color: C.red, size: 20 })],
        }),
        para("当前问题：搜索 API 不支持 offset 参数，前端分页 UI 存在但点击上/下页只是重新发送相同请求，永远显示第一页的相同结果。用户无法浏览“第 2 页”以后的内容。"),
        bullet("优化：后端 /search 接口增加 offset 参数，前端 goPage() 将 offset = (page-1)*limit 传入请求"),
        bullet("优先级：当知识库达到千级规模时将成为用户必见的体验断点"),

        // U2: No learning history
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("U2 无学习历史记录", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：用户每次访问都是“全新状态”，无搜索历史、无收藏、无学习记录。学完关页就全丢了，无法形成学习闭环。"),
        bullet("优化短期：用 localStorage 存储最近 20 条搜索记录 + 收藏列表，在页面底部或侧边栏展示"),
        bullet("优化中期：增加用户系统（快速登录），持久化学习轨迹、掌握程度、复习计划"),

        // U3: No mobile adaptation
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("U3 移动端适配不足", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：仅 @media(max-width:600px) 一条响应式规则（只调小了字号），实际在手机上筛选栏会溢出、示例标签折行凌乱、视频播放器可能超出屏幕。"),
        bullet("优化：筛选栏在 ≤480px 时改为单列堆叠；示例标签改为可横向滚动；视频播放器 100vw 全屏"),
        bullet("用户场景：英语学习大量发生在通勤/磨耳场景，移动端是核心设备"),

        // U4: Video auto-expand
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("U4 视频切片默认展开撤去了折叠控制", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：所有有视频的卡片都默认展开播放器（display:block），“▶️ 播放”按钮和 toggleVP() 已被删除。当返回 10+ 有视频的结果时，页面会变得非常长，用户需要大量滚动才能浏览全部结果。每个视频占据 16:9 的宽幅，让英文台词本身被“挤到后面”。"),
        bullet("优化：恢复“折叠”设计 —— 默认只显示缩略图，点击缩略图或 ▶️ 按钮展开播放器"),
        bullet("方案参考：首条结果自动展开，其余折叠（节省页面空间，引导注意力）"),

        // U5: Debug info in production
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("U5 原始调试信息暴露给用户", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：每张卡片底部显示“质量分 0.637 · 出现 3 次 · 综合分 0.721”，这些参数对学习者毫无意义，反而引发困惑和干扰。同时“相似度 87%”虽然有值但显示位置过于突兀。"),
        bullet("优化：移除 quality_score、occurrence_count、rank_score 展示，仅保留相似度（考虑改为“匹配度”时展示为星级：★★★★☆）"),
        bullet("调试信息可保留在“思考摘要卡”内部，供开发者展开查看"),

        // U6: No keyboard navigation
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("U6 缺少键盘快捷键", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：仅 Enter 触发搜索，无其他快捷键支持。用户不能用↑↓切换结果、Esc 清空搜索、/ 聚焦搜索框。"),
        bullet("优化：添加 / 聚焦搜索框、Esc 清空、↑↓ 切换结果高亮、C 复制当前高亮台词"),

        // U7: header links
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("U7 Header 链接布局不当", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：顶部导航栏显示“⚙️ 管理后台”和“\u{1F4E1} API”链接，但这是学习产品用户页，暴露后台入口不合适。应替换为学习相关入口（如“我的收藏”“学习记录”）。"),
        bullet("优化：将“管理后台”“API”移至页脚小字链接，顶部放置用户功能入口"),

        // ── 2.2.2 Learning convenience ──
        heading("2.2.2 学习便捷性", HeadingLevel.HEADING_3),

        // L1: No pronunciation
        new Paragraph({
          spacing: { before: 160, after: 80 },
          children: [bold("L1 缺少发音学习支持", { color: C.red, size: 22 }), run(" —— P0", { color: C.red, size: 20 })],
        }),
        para("当前问题：英文台词只有文字，没有音标、没有发音按钮。英语学习的核心说的“听说读写”中“听”和“说”完全缺失。即使有视频切片可以听原音，但未附视频的台词完全无法获取发音。"),
        bullet("优化短期：在英文台词旁增加 🔊 TTS 按钮（使用浏览器原生 SpeechSynthesis API，零成本）"),
        bullet("优化中期：集成优质 TTS 服务（如 Edge TTS），提供美/英发音切换"),

        // L2: No context learning
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("L2 缺少语境化学习引导", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：卡片只展示“台词 + 翻译”，缺少“怎么用”的引导。用户看完台词不知道什么情况用、有什么语用注意点、可以如何替换学习。"),
        bullet("优化：在卡片下方增加可展开的“学习笔记”区域："),
        bullet("语境说明：“这句话通常用在…场景中，语气偶尔/正式…”", "bullets2"),
        bullet("举一反三：“相似表达：...”“反义表达：...”", "bullets2"),
        bullet("填空练习：“I really ___ what you did for me.”", "bullets2"),

        // L3: No spaced repetition
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("L3 缺少间隔重复 / 复习机制", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：用户查完就走，无复习提醒、无练习测试。学习科学证明“检索”只是输入的第一步，“复习+练习”才是记住的关键。"),
        bullet("优化：增加“收藏”功能 + 简易闪卡复习模式（要求用户回忘中文含义或英文台词）"),
        bullet("未来：接入 SM-2 算法做美式间隔重复"),

        // L4: Sort options not learner-friendly
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("L4 排序选项非学习者视角", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：排序选项为“相似度优先 / 综合排序 / 质量分”，这是开发者视角的命名。对学习者而言，“质量分”不知所云。"),
        bullet("优化：改为“最相关 / 最地道（质量优先） / 由易到难”"),

        // ── 2.2.3 Learning effectiveness ──
        heading("2.2.3 学习有效性", HeadingLevel.HEADING_3),

        // E1: No progress tracking
        new Paragraph({
          spacing: { before: 160, after: 80 },
          children: [bold("E1 无学习进度感知", { color: C.red, size: 22 }), run(" —— P0", { color: C.red, size: 20 })],
        }),
        para("当前问题：用户无法知道“我学了多少”“还有多少值得学”“我的水平在哪”。缺乏进度感知直接导致用户流失——这是英语学习产品的核心留存指标。"),
        bullet("优化：增加“学习仪表盘”——今日学习词数、连续学习天数、已掌握/待复习数量"),
        bullet("参考产品：Anki、多邻国、不背单词的“打卡”机制"),

        // E2: No difficulty progression
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("E2 难度递进路径缺失", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：难度筛选存在但是被动的。系统不会根据用户历史推荐“你已掌握这个难度，试试更难的”。初学者可能上来就看到充满促词/强调的高难度台词而受挫。"),
        bullet("优化：根据用户历史搜索/收藏的难度分布，自动调整默认难度筛选"),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════
        // 3. ADMIN PANEL EVALUATION
        // ════════════════════════════════════════════
        heading("三、管理后台测评", HeadingLevel.HEADING_1),

        heading("3.1 亮点与优势", HeadingLevel.HEADING_2),
        bullet("功能覆盖全面：从字幕上传、TED 链接获取、知识库管理、视频切片、任务队列到统计面板，形成完整的内容生产闭环"),
        bullet("多格式字幕支持：.srt / .ass / .ssa，支持单语单传、双语对齐、TED 自动获取三种模式"),
        bullet("任务队列可视化：字幕获取 / 字幕入库 / 视频切片 / 知识库管理四类任务独立展示，支持自动刷新和状态筛选"),
        bullet("切片卡片化设计：视频切片列表已从表格升级为卡片网格，缩略图 + 状态徽章 + 置信度覆盖，视觉效果良好"),
        bullet("批量操作完善：全选/多选/批量删除/批量打标/批量修改/批量评估，大规模运营效率有保障"),
        bullet("可折叠面板：重新打标 / 合并相邻分句等高级功能分组在可折叠面板内，降低页面复杂度"),

        heading("3.2 问题与优化建议", HeadingLevel.HEADING_2),

        // A1: sidebar too deep
        new Paragraph({
          spacing: { before: 160, after: 80 },
          children: [bold("A1 侧边栏导航层级混乱，学习曲线陡峭", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：侧边栏有 15 个菜单项，分为“内容管理 / 视频处理 / 任务队列 / 系统”四组，但功能边界模糊：“字幕上传”和“字幕处理”是什么关系？“字幕记录”和“知识库”有什么区别？新用户需要一个个点开才能理解。"),
        bullet("优化：合并同类菜单，提供“引导式工作流”"),
        bullet("“字幕上传”+“字幕处理”→ 合并为“字幕入库”一个菜单项", "bullets2"),
        bullet("“字幕记录”→ 移入“知识库管理”作为子 Tab", "bullets2"),
        bullet("四个任务队列→ 合并为一个“任务队列”页面，用 Tab 切换任务类型", "bullets2"),

        // A2: no onboarding
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("A2 缺少新手引导 / 快速开始", { color: C.orange, size: 22 }), run(" —— P1", { color: C.orange, size: 20 })],
        }),
        para("当前问题：新用户进入后台后面对 15 个菜单项，没有任何引导。要成功上传并处理一套字幕，需要理解“字幕上传 → 字幕记录 → 同步知识库 → 上传视频 → 切片”的完整链路。"),
        bullet("优化：增加“Quick Start”卡片，在首页显示“1. 上传字幕 → 2. 审核确认 → 3. 上传视频 → 4. 查看效果”步骤引导"),

        // A3: stats too minimal
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("A3 统计面板过于基础", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：数据卡片只展示数字汇总，无时间趋势、无转化漏斗、无搜索热词。运营者无法回答“本周新增了多少知识节点”“哪类台词被搜索最多”等关键运营问题。"),
        bullet("优化：增加搜索日志 + 热词榜 + 时间趋势图（字幕输入量/知识节点增量/搜索量）"),

        // A4: No auth
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("A4 后台无登录保护", { color: C.red, size: 22 }), run(" —— P0", { color: C.red, size: 20 })],
        }),
        para("当前问题：访问 /admin 无需任何认证，任何人都可以进入后台删除知识库、修改配置。config.py 配置了 ADMIN_USER / ADMIN_PASS 但前端未接入。"),
        bullet("优化：实现 Basic Auth 或 session 认证，前端加登录弹窗，未登录状态下后台 API 返回 401"),

        // A5: no error recovery
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("A5 批量操作缺少确认及撤回", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：“一键清空视频和切片”“批量删除”等破坏性操作仅有简单 confirm()，无输入确认，无撤回机制。误点一下可能损失大量数据。"),
        bullet("优化：批量删除 > 10 条时要求输入确认文字；一键清空等超高风险操作加“输入剧名确认”门槛"),

        // A6: knowledge table could be better
        new Paragraph({
          spacing: { before: 200, after: 80 },
          children: [bold("A6 知识库表格数据密度高，可读性待提升", { color: C.blue, size: 22 }), run(" —— P2", { color: C.blue, size: 20 })],
        }),
        para("当前问题：知识库表格有11列数据，在普通屏幕上必须横向滚动。难度/场景/情绪三列可以合并为综合标签列。"),
        bullet("优化：难度+场景+情绪合并为一列多徽章；操作列固定在右侧；考虑参考切片列表的卡片化设计"),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════
        // 4. ISSUE MATRIX
        // ════════════════════════════════════════════
        heading("四、问题清单总览", HeadingLevel.HEADING_1),

        new Table({
          width: { size: CONTENT_W, type: WidthType.DXA },
          columnWidths: [500, 600, 1200, 2200, 4860],
          rows: [
            // Header
            new TableRow({
              children: ["#", "优先级", "模块", "问题", "概述"].map((t, i) =>
                new TableCell({
                  borders,
                  margins: cellMargins,
                  width: { size: [500, 600, 1200, 2200, 4860][i], type: WidthType.DXA },
                  shading: { fill: C.brand, type: ShadingType.CLEAR },
                  children: [new Paragraph({ children: [bold(t, { size: 20, color: C.white })] })],
                })
              ),
            }),
            issueRow("U1", "P0", "用户页", "分页失效", "搜索 API 不支持 offset，分页 UI 形同虚设"),
            issueRow("L1", "P0", "用户页", "无发音支持", "英文台词无 TTS，“听说”体验缺失"),
            issueRow("E1", "P0", "用户页", "无进度感知", "无学习仪表盘、无打卡机制"),
            issueRow("A4", "P0", "后台", "无登录保护", "config 有账密但前端未接入"),
            issueRow("U2", "P1", "用户页", "无学习历史", "无搜索历史、无收藏、无学习记录"),
            issueRow("U3", "P1", "用户页", "移动端不足", "响应式规则仅一条，实际手机体验差"),
            issueRow("L2", "P1", "用户页", "无语境学习", "只展示台词+翻译，缺“怎么用”引导"),
            issueRow("L3", "P1", "用户页", "无复习机制", "查完就走，无闪卡/测试/重复学习"),
            issueRow("E2", "P1", "用户页", "难度递进缺失", "系统不跟据历史调整推荐难度"),
            issueRow("A1", "P1", "后台", "导航混乱", "15个菜单项，功能边界模糊"),
            issueRow("A2", "P1", "后台", "无新手引导", "新用户无法理解完整工作流"),
            issueRow("U4", "P2", "用户页", "视频默认展开", "多视频结果页面过长，卡片被视频挤压"),
            issueRow("U5", "P2", "用户页", "调试信息泄露", "质量分/综合分等参数对用户无意义"),
            issueRow("U6", "P2", "用户页", "无快捷键", "缺少 / 聚焦、Esc 清空、↑↓切换等"),
            issueRow("U7", "P2", "用户页", "Header 链接不当", "学习页暴露后台/API入口"),
            issueRow("L4", "P2", "用户页", "排序命名不友好", "“质量分”等开发者术语"),
            issueRow("A3", "P2", "后台", "统计过基础", "无时间趋势、无搜索热词"),
            issueRow("A5", "P2", "后台", "批量操作风险", "破坏性操作无输入确认门槛"),
            issueRow("A6", "P2", "后台", "知识库可读性", "11列表格需横向滚动"),
          ],
        }),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════
        // 5. ROADMAP
        // ════════════════════════════════════════════
        heading("五、建议优化路线图", HeadingLevel.HEADING_1),

        heading("Phase 1：基础体验修复（建议 1-2 天）", HeadingLevel.HEADING_2),
        para("目标：修复功能性问题，让产品基本可用。"),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("U1 修复分页", { size: 22 }), run("：后端 /search 增加 offset 参数，前端传递页码", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("A4 后台登录保护", { size: 22 }), run("：接入已配置的 ADMIN_USER/PASS，前端加登录弹窗", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("U5 移除调试信息", { size: 22 }), run("：前端移除 quality_score/rank_score/occurrence 显示", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("U4 视频默认折叠", { size: 22 }), run("：恢复缩略图+点击展开模式", { size: 22 })],
        }),

        heading("Phase 2：学习核心体验（建议 3-5 天）", HeadingLevel.HEADING_2),
        para("目标：完成“搜索 → 学习 → 复习”最小闭环。"),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("L1 TTS 发音", { size: 22 }), run("：使用浏览器 SpeechSynthesis API，零成本实现台词朗读", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("U2 搜索历史+收藏", { size: 22 }), run("：localStorage 存储最近搜索 + 收藏列表", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("L3 简易闪卡复习", { size: 22 }), run("：收藏列表 + 翻转卡片复习模式", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("E1 学习仪表盘", { size: 22 }), run("：今日学习/连续天数/已掌握量计数卡片", { size: 22 })],
        }),

        heading("Phase 3：体验精细化（建议 3-5 天）", HeadingLevel.HEADING_2),
        para("目标：提升移动端体验 + 运营效率。"),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("U3 移动端适配", { size: 22 }), run("：完善响应式布局，重点适配 375-480px 屏幕", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("L2 语境化学习引导", { size: 22 }), run("：增加“学习笔记”区域（AI 生成用法说明 + 举一反三 + 填空练习）", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("A1 导航简化", { size: 22 }), run("：15 项 → 8 项，同类合并 + 快速开始引导", { size: 22 })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 80 },
          children: [bold("A3 统计增强", { size: 22 }), run("：增加搜索日志、热词榜、时间趋势图", { size: 22 })],
        }),

        heading("Phase 4：深度学习体验（建议 1-2 周）", HeadingLevel.HEADING_2),
        para("目标：构建用户系统，实现个性化学习。"),
        bullet("用户注册/登录系统（支持微信扫码）"),
        bullet("SM-2 算法间隔重复，智能推送复习提醒"),
        bullet("E2 难度递进——根据学习历史自动调整推荐难度"),
        bullet("学习成就体系：每日打卡 / 场景全解锁 / 剧集收藏册"),

        spacer(300),

        // ── closing ──
        new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: C.lightPurple } },
          spacing: { before: 200, after: 100 },
          children: [],
        }),
        para("DramaLingo 的核心理念“用影视台词学地道英语”既有新意又有实用价值，AI 多信号检索技术底座扮实。当前 MVP 最紧迫的不是技术加强，而是从“工具”转向“学习产品”的体验闭环补全。只有让用户感受到“我在进步”，才能实现从“开一次”到“每天来”的跳跃。", { run: { italics: true, color: C.muted } }),
      ],
    },
  ],
});

// ── Output ──
const outPath = "D:\\Dramalingo_mvp\\Dramalingo_mvp-main\\DramaLingo_MVP_产品测评报告.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log("Report generated:", outPath);
});
