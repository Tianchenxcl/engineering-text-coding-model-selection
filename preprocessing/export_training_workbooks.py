from pathlib import Path
import argparse,csv,json
from openpyxl import Workbook

def save(path,records,columns,sheet):
    wb=Workbook(write_only=True);ws=wb.create_sheet(sheet);ws.append(columns)
    for row in records:ws.append([row.get(k) for k in columns])
    path.parent.mkdir(parents=True,exist_ok=True);wb.save(path)

def main():
    parser=argparse.ArgumentParser(description='Convert private intermediate pairs to the training workbook schema using openpyxl.')
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.package_root
    files=sorted((root/'_build_intermediate/supervised_pairs').glob('pair*.json'))
    if not files:raise FileNotFoundError('Private intermediate pairs are required. Run prepare_hy_tmp_package_0713.py first.')
    for p in files:
        d=json.loads(p.read_text(encoding='utf-8-sig'))
        for split in ['train','test']:save(root/'supervised_data'/d[split+'_file'],d[split],d['columns'],'Data')
    with (root/'_build_intermediate/task_file_mapping.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);records=list(reader);columns=reader.fieldnames
    save(root/'manifests/task_file_mapping.xlsx',records,columns,'TaskMapping')
    print(f'Wrote {len(files)} pairs and task mapping. Workbook formatting differs from the original exporter; values and column order are preserved.')

if __name__=='__main__':main()
