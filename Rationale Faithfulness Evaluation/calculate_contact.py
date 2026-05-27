import os
from Bio.PDB import PDBParser, Selection
from Bio.PDB.NeighborSearch import NeighborSearch
import pandas as pd

# 设定阈值：小于 5.0 埃 被认为是接触 (Contact)
CONTACT_THRESHOLD = 5.0

# PDB文件本地目录
PDB_DIR = r"D:\Desktop\test_antigenHLAI\Data\crystal_structure\pdb"

def get_pdb_file_path(pdb_id):
    """获取本地PDB文件路径"""
    # 尝试不同的文件扩展名
    for ext in ['.pdb', '.ent', '']:
        file_path = os.path.join(PDB_DIR, f"{pdb_id}{ext}")
        if os.path.exists(file_path):
            return file_path
    # 如果没找到，尝试小写
    for ext in ['.pdb', '.ent', '']:
        file_path = os.path.join(PDB_DIR, f"{pdb_id.lower()}{ext}")
        if os.path.exists(file_path):
            return file_path
    
    print(f"Warning: PDB file not found for {pdb_id} in {PDB_DIR}")
    return None

def get_interacting_residues(pdb_file, pep_chain_id, tcr_chain_id):
    """计算肽段和TCR之间的接触残基"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    model = structure[0]
    
    try:
        pep_chain = model[pep_chain_id]
        tcr_chain = model[tcr_chain_id]
    except KeyError:
        print(f"Warning: Chain not found in {pdb_file}. Available chains: {[c.id for c in model]}")
        return None, None
    
    # 获取两条链的所有原子
    pep_atoms = Selection.unfold_entities(pep_chain, 'A')
    tcr_atoms = Selection.unfold_entities(tcr_chain, 'A')
    
    # 构建TCR原子的KD-Tree进行快速近邻搜索
    ns = NeighborSearch(tcr_atoms)
    
    interacting_pep = set()
    interacting_tcr = set()
    
    for p_atom in pep_atoms:
        close_tcr_atoms = ns.search(p_atom.coord, CONTACT_THRESHOLD)
        if close_tcr_atoms:
            p_res = p_atom.get_parent()
            interacting_pep.add((p_res.get_resname(), p_res.get_id()[1]))
            for t_atom in close_tcr_atoms:
                t_res = t_atom.get_parent()
                interacting_tcr.add((t_res.get_resname(), t_res.get_id()[1]))
    
    return interacting_pep, interacting_tcr

def parse_pdbid(pdbid_string):
    """
    解析pdbid字符串，如 "8gom_C_A_E" 
    返回: (pdb_id, pep_chain, hla_chain, tcr_chain)
    """
    parts = pdbid_string.split('_')
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    else:
        raise ValueError(f"Invalid pdbid format: {pdbid_string}. Expected format: pdbid_pep_hla_tcr")

def main(csv_file):
    """
    从CSV文件读取数据并处理
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 检查必要的列是否存在
    required_columns = ['pdbid', 'pep_seq', 'cdr3_seq']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    results = []
    
    # 遍历每一行数据
    for idx, row in df.iterrows():
        pdbid_full = row['pdbid']
        
        try:
            # 解析pdbid
            pdb_id, pep_c, hla_c, tcr_c = parse_pdbid(pdbid_full)
            
            # 获取本地PDB文件路径
            pdb_file = get_pdb_file_path(pdb_id)
            if not pdb_file:
                continue
            
            # 计算接触残基
            pep_contacts, tcr_contacts = get_interacting_residues(pdb_file, pep_c, tcr_c)
            
            if pep_contacts and tcr_contacts:
                print(f"\n--- PDB: {pdbid_full.upper()} ---")
                pep_contact_list = sorted([f'{r[0]}{r[1]}' for r in pep_contacts])
                tcr_contact_list = sorted([f'{r[0]}{r[1]}' for r in tcr_contacts])
                
                print(f"Peptide Contact Residues: {pep_contact_list}")
                print(f"TCR Contact Residues:     {tcr_contact_list}")
                print(f"Peptide Sequence: {row['pep_seq']}")
                print(f"CDR3 Sequence: {row['cdr3_seq']}")
                
                # 保存结果
                results.append({
                    'pdbid': pdbid_full,
                    'pep_seq': row['pep_seq'],
                    'cdr3_seq': row['cdr3_seq'],
                    'pep_contacts': ', '.join(pep_contact_list),
                    'tcr_contacts': ', '.join(tcr_contact_list),
                    'pep_contact_count': len(pep_contacts),
                    'tcr_contact_count': len(tcr_contacts)
                })
            else:
                print(f"\n--- PDB: {pdbid_full.upper()} ---")
                print("No interacting residues found or chain not found")
                
        except Exception as e:
            print(f"\n--- Error processing {row['pdbid']}: {e} ---")
            continue
    
    # 将结果保存为新的CSV文件
    if results:
        results_df = pd.DataFrame(results)
        output_file = csv_file.replace('.csv', '_contacts.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # 指定你的CSV文件路径
    csv_file = r"D:\Desktop\test_antigenHLAI\Data\crystal_structure\info_noredudant.csv"  # 修改为你的CSV文件名
    main(csv_file)