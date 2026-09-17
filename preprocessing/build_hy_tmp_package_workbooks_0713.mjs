import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const here = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.dirname(here);
const pairRoot = path.join(packageRoot, "_build_intermediate", "supervised_pairs");
const mappingCsv = path.join(packageRoot, "_build_intermediate", "task_file_mapping.csv");
const dataRoot = path.join(packageRoot, "supervised_data");
const manifestRoot = path.join(packageRoot, "manifests");
const qaRoot = path.join(packageRoot, "_qa");

const colors = {
  navy: "#1F4E78",
  white: "#FFFFFF",
  text: "#1F1F1F",
  line: "#E7E6E6",
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
  if (value === "text") return 76;
  if (value.includes("selected_labels") || value.includes("rule")) return 48;
  if (value.includes("design_tier") || value.includes("class_balance_design")) return 30;
  if (value.includes("samples_per_label")) return 34;
  if (value.includes("sample_id") || value.includes("task_id") || value.includes("workbook")) return 38;
  if (value.includes("source_row_ids")) return 30;
  if (value.includes("label_name") || value.includes("file")) return 26;
  return 18;
}

function recordsMatrix(records, columns) {
  return [columns, ...records.map((row) => columns.map((column) => row[column] ?? null))];
}

function formatDataSheet(sheet, matrix) {
  const rowCount = matrix.length;
  const colCount = matrix[0].length;
  const used = sheet.getRangeByIndexes(0, 0, rowCount, colCount);
  used.values = matrix;
  used.format.font = { name: "Aptos", size: 9, color: colors.text };
  used.format.verticalAlignment = "top";
  used.format.borders = { preset: "insideHorizontal", style: "thin", color: colors.line };
  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  header.format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 30;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  for (let col = 0; col < colCount; col += 1) {
    const column = sheet.getRangeByIndexes(0, col, rowCount, 1);
    const width = widthForHeader(matrix[0][col]);
    column.format.columnWidth = width;
    if (matrix[0][col] !== "text") column.format.wrapText = false;
    if (matrix[0][col] === "text") column.format.wrapText = false;
  }
}

async function exportWorkbook(workbook, outputPath) {
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
}

async function buildDataWorkbook(records, columns, outputPath) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Data");
  formatDataSheet(sheet, recordsMatrix(records, columns));
  await exportWorkbook(workbook, outputPath);
  return workbook;
}

async function renderRepresentative(workbook, fileName) {
  await fs.mkdir(qaRoot, { recursive: true });
  const blob = await workbook.render({ sheetName: "Data", range: "A1:M16", scale: 1, format: "png" });
  await fs.writeFile(path.join(qaRoot, fileName), new Uint8Array(await blob.arrayBuffer()));
}

async function buildBatch(start, end) {
  await fs.mkdir(dataRoot, { recursive: true });
  const files = (await fs.readdir(pairRoot)).filter((name) => name.endsWith(".json")).sort();
  const selected = files.slice(start, Math.min(end, files.length));
  for (let i = 0; i < selected.length; i += 1) {
    const payload = JSON.parse(await fs.readFile(path.join(pairRoot, selected[i]), "utf8"));
    const trainWorkbook = await buildDataWorkbook(
      payload.train,
      payload.columns,
      path.join(dataRoot, payload.train_file),
    );
    await buildDataWorkbook(payload.test, payload.columns, path.join(dataRoot, payload.test_file));
    if (start === 0 && i === 0) await renderRepresentative(trainWorkbook, "train1_preview.png");
    if (global.gc) global.gc();
  }
  console.log(JSON.stringify({ start, end, builtPairs: selected.length, totalPairs: files.length }));
}

async function buildMappingWorkbook() {
  await fs.mkdir(manifestRoot, { recursive: true });
  const csvText = await fs.readFile(mappingCsv, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "TaskMapping" });
  const sheet = workbook.worksheets.getItem("TaskMapping");
  const used = sheet.getUsedRange();
  const matrix = used.values;
  formatDataSheet(sheet, matrix);
  const outputPath = path.join(manifestRoot, "task_file_mapping.xlsx");
  const inspected = await workbook.inspect({
    kind: "table",
    range: "TaskMapping!A1:Z8",
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 26,
    maxChars: 5000,
  });
  await fs.mkdir(qaRoot, { recursive: true });
  await fs.writeFile(path.join(qaRoot, "mapping_inspect.ndjson"), inspected.ndjson, "utf8");
  const preview = await workbook.render({ sheetName: "TaskMapping", range: "A1:J12", scale: 1, format: "png" });
  await fs.writeFile(path.join(qaRoot, "task_mapping_preview.png"), new Uint8Array(await preview.arrayBuffer()));
  await exportWorkbook(workbook, outputPath);
  console.log(outputPath);
}

const [mode = "help", arg1, arg2] = process.argv.slice(2);
if (mode === "batch") {
  await buildBatch(Number(arg1 ?? 0), Number(arg2 ?? Number.MAX_SAFE_INTEGER));
} else if (mode === "mapping") {
  await buildMappingWorkbook();
} else {
  console.log("Usage: node build_hy_tmp_package_workbooks_0713.mjs batch <start> <end> | mapping");
}
