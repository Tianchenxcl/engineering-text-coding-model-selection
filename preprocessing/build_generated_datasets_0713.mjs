import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "generated_datasets");
const intermediate = path.join(root, "intermediate_json");
const processed = path.join(here, "processed");
const manifestsCsv = path.join(root, "manifests_csv");
const qaDir = path.join(root, "qa_previews");

const outputDirs = {
  master: path.join(root, "01_master"),
  common: path.join(root, "02_common24"),
  supervised: path.join(root, "03_supervised_tasks"),
  llm: path.join(root, "04_llm_tasks"),
  manifests: path.join(root, "05_manifests"),
};

const colors = {
  navy: "#1F4E78",
  lightBlue: "#D9EAF7",
  paleBlue: "#EEF5FA",
  line: "#D9E2F3",
  white: "#FFFFFF",
  text: "#1F1F1F",
};

function columnLetter(index) {
  let n = index + 1;
  let output = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    n = Math.floor((n - 1) / 26);
  }
  return output;
}

function widthForHeader(header) {
  const value = String(header).toLowerCase();
  if (value === "text" || value.includes("normalized_text")) return 70;
  if (value.includes("description") || value.includes("template") || value.includes("prompt_components")) return 50;
  if (value.includes("selected_labels") || value.includes("samples_per_label")) return 42;
  if (value.includes("rule") || value.includes("source_row_ids")) return 38;
  if (value.includes("design_tier") || value.includes("anchor_condition")) return 34;
  if (value.includes("model_version")) return 32;
  if (value.includes("prompt_strategy")) return 24;
  if (value.includes("task_id") || value.includes("combination_id") || value.includes("sample_id")) return 32;
  if (value.includes("label_name") || value.includes("selection_reason")) return 26;
  return 18;
}

function isRateHeader(header) {
  const value = String(header).toLowerCase();
  return value.includes("rate") || value.includes("similarity") || value.includes("imbalance");
}

function recordsToMatrix(records) {
  if (!records?.length) return [["No records"]];
  const headers = Object.keys(records[0]);
  const rows = records.map((record) => headers.map((header) => record[header] ?? null));
  return [headers, ...rows];
}

function metadataToMatrix(metadata) {
  return [
    ["Field", "Value"],
    ...Object.entries(metadata).map(([key, value]) => [
      key,
      value !== null && typeof value === "object" ? JSON.stringify(value) : (value ?? null),
    ]),
  ];
}

function formatDataSheet(sheet, matrix, options = {}) {
  const rowCount = matrix.length;
  const colCount = matrix[0]?.length ?? 1;
  const used = sheet.getRangeByIndexes(0, 0, rowCount, colCount);
  used.values = matrix;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Aptos", size: 9, color: colors.text };
  used.format.verticalAlignment = "top";
  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  header.format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 32;
  used.format.borders = { preset: "insideHorizontal", style: "thin", color: "#E7E6E6" };
  const headers = matrix[0];
  for (let col = 0; col < colCount; col += 1) {
    const colRange = sheet.getRangeByIndexes(0, col, rowCount, 1);
    const width = widthForHeader(headers[col]);
    colRange.format.columnWidth = width;
    if (width >= 32) colRange.format.wrapText = true;
    if (isRateHeader(headers[col]) && rowCount > 1) {
      sheet.getRangeByIndexes(1, col, rowCount - 1, 1).format.numberFormat = "0.0000";
    }
  }
  if (options.metadata && rowCount > 1) {
    sheet.getRangeByIndexes(1, 0, rowCount - 1, 1).format = {
      fill: colors.paleBlue,
      font: { bold: true, color: colors.text },
      wrapText: true,
    };
    sheet.getRange("A1:A100").format.columnWidth = 40;
    sheet.getRange("B1:B100").format.columnWidth = 72;
    sheet.getRange("B1:B100").format.wrapText = true;
  }
}

function addRecordsSheet(workbook, name, records, options = {}) {
  const sheet = workbook.worksheets.add(name);
  const matrix = Array.isArray(records) ? recordsToMatrix(records) : metadataToMatrix(records);
  formatDataSheet(sheet, matrix, options);
  return sheet;
}

async function renderWorkbook(workbook, prefix) {
  await fs.mkdir(qaDir, { recursive: true });
  for (const sheet of workbook.worksheets.items) {
    const used = sheet.getUsedRange();
    const values = used.values;
    const rows = Math.min(values.length, 20);
    const cols = Math.min(values[0]?.length ?? 1, 12);
    const range = `A1:${columnLetter(cols - 1)}${rows}`;
    const preview = await workbook.render({ sheetName: sheet.name, range, scale: 1, format: "png" });
    const fileName = `${prefix}_${sheet.name.replace(/[^A-Za-z0-9_-]/g, "_")}.png`;
    await fs.writeFile(path.join(qaDir, fileName), new Uint8Array(await preview.arrayBuffer()));
  }
}

async function exportWorkbook(workbook, outputPath) {
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
}

async function buildSupervisedFile(filePath, outputPath, render = false) {
  const payload = JSON.parse(await fs.readFile(filePath, "utf8"));
  const workbook = Workbook.create();
  addRecordsSheet(workbook, "Metadata", payload.metadata, { metadata: true });
  addRecordsSheet(workbook, "DevelopmentPool", payload.development_pool);
  addRecordsSheet(workbook, "FixedTest", payload.fixed_test);
  addRecordsSheet(workbook, "LabelMapping", payload.label_mapping);
  if (render) await renderWorkbook(workbook, "representative_supervised");
  await exportWorkbook(workbook, outputPath);
}

async function buildCommonFile(filePath, outputPath, render = false) {
  const payload = JSON.parse(await fs.readFile(filePath, "utf8"));
  const workbook = Workbook.create();
  addRecordsSheet(workbook, "Metadata", payload.metadata, { metadata: true });
  addRecordsSheet(workbook, "FixedTest", payload.fixed_test);
  addRecordsSheet(workbook, "Codebook", payload.codebook);
  if (render) await renderWorkbook(workbook, "representative_common24");
  await exportWorkbook(workbook, outputPath);
}

async function buildLlmFile(filePath, outputPath, render = false) {
  const payload = JSON.parse(await fs.readFile(filePath, "utf8"));
  const workbook = Workbook.create();
  addRecordsSheet(workbook, "Metadata", payload.metadata, { metadata: true });
  addRecordsSheet(workbook, "FixedTest", payload.fixed_test);
  addRecordsSheet(workbook, "Demonstrations", payload.demonstrations);
  addRecordsSheet(workbook, "Codebook", payload.codebook);
  addRecordsSheet(workbook, "PromptTemplates", payload.prompt_templates);
  addRecordsSheet(workbook, "RunPlan", payload.run_plan);
  if (render) await renderWorkbook(workbook, "representative_llm");
  await exportWorkbook(workbook, outputPath);
}

async function csvValues(filePath, sheetName = "Imported") {
  const text = await fs.readFile(filePath, "utf8");
  const temporary = await Workbook.fromCSV(text, { sheetName });
  return temporary.worksheets.getItem(sheetName).getUsedRange().values;
}

async function addCsvSheet(workbook, sheetName, filePath) {
  const values = await csvValues(filePath);
  const sheet = workbook.worksheets.add(sheetName);
  formatDataSheet(sheet, values);
  return sheet;
}

async function buildMasterAndManifests() {
  await Promise.all(Object.values(outputDirs).map((directory) => fs.mkdir(directory, { recursive: true })));

  const master = Workbook.create();
  const generationSummary = JSON.parse(await fs.readFile(path.join(root, "generation_summary.json"), "utf8"));
  addRecordsSheet(master, "Summary", generationSummary, { metadata: true });
  await addCsvSheet(master, "TrainingPool", path.join(processed, "deduplicated_training_pool.csv"));
  await addCsvSheet(master, "FixedTest991", path.join(processed, "fixed_test_991.csv"));
  await addCsvSheet(master, "LabelCodebook", path.join(manifestsCsv, "label_codebook.csv"));
  await addCsvSheet(master, "DemoBank44", path.join(manifestsCsv, "llm_demonstration_bank_44.csv"));
  await renderWorkbook(master, "master_data");
  await exportWorkbook(master, path.join(outputDirs.master, "master_data.xlsx"));

  const manifestBooks = [
    {
      output: "supervised_task_manifest_240.xlsx",
      sheets: [
        ["TaskManifest", "supervised_task_manifest_240.csv"],
        ["SelectedDesign", "selected_supervised_task_design_240.csv"],
        ["LabelCodebook", "label_codebook.csv"],
      ],
    },
    {
      output: "common24_manifest.xlsx",
      sheets: [
        ["Common24", "common24_manifest.csv"],
        ["LabelCodebook", "label_codebook.csv"],
      ],
    },
    {
      output: "llm_manifest_40_combinations_360_runs.xlsx",
      sheets: [
        ["LLM40", "llm_combination_manifest_40.csv"],
        ["RunPlan360", "llm_run_plan_360.csv"],
        ["DemoBank44", "llm_demonstration_bank_44.csv"],
        ["LabelCodebook", "label_codebook.csv"],
      ],
    },
  ];

  for (const book of manifestBooks) {
    const workbook = Workbook.create();
    for (const [sheetName, fileName] of book.sheets) {
      await addCsvSheet(workbook, sheetName, path.join(manifestsCsv, fileName));
    }
    await renderWorkbook(workbook, path.parse(book.output).name);
    await exportWorkbook(workbook, path.join(outputDirs.manifests, book.output));
  }
}

async function buildBatch(kind, start, end) {
  const inputFolder = path.join(intermediate, kind);
  const files = (await fs.readdir(inputFolder))
    .filter((name) => name.endsWith(".json"))
    .sort();
  const selected = files.slice(start, Math.min(end, files.length));
  const outputFolder = outputDirs[kind === "common24" ? "common" : kind];
  await fs.mkdir(outputFolder, { recursive: true });
  for (let index = 0; index < selected.length; index += 1) {
    const fileName = selected[index];
    const inputPath = path.join(inputFolder, fileName);
    const outputPath = path.join(outputFolder, `${path.parse(fileName).name}.xlsx`);
    const render = start === 0 && index === 0;
    if (kind === "supervised") await buildSupervisedFile(inputPath, outputPath, render);
    if (kind === "common24") await buildCommonFile(inputPath, outputPath, render);
    if (kind === "llm") await buildLlmFile(inputPath, outputPath, render);
    if (global.gc) global.gc();
  }
  console.log(JSON.stringify({ kind, start, end, built: selected.length, total: files.length }));
}

const [mode = "help", arg1, arg2, arg3] = process.argv.slice(2);
if (mode === "manifests") {
  await buildMasterAndManifests();
  console.log("Master and manifest workbooks created.");
} else if (mode === "batch") {
  const kind = arg1;
  const start = Number(arg2 ?? 0);
  const end = Number(arg3 ?? Number.MAX_SAFE_INTEGER);
  if (!["supervised", "common24", "llm"].includes(kind)) throw new Error(`Unknown batch kind: ${kind}`);
  await buildBatch(kind, start, end);
} else {
  console.log("Usage: node build_generated_datasets_0713.mjs manifests | batch <supervised|common24|llm> <start> <end>");
}
