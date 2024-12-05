import json
import os
import os.path as osp
import sys
import time
import datetime
import csv
import shutil
import signal
import subprocess

import numpy as np
import pandas as pd

from tokenizer import code_tokenize


# Require joern v0.2.5
PROJECT_DIR = osp.realpath(osp.join(osp.dirname(__file__), '..', '..'))
JOERN_DIR = osp.join(PROJECT_DIR, 'joern')

NODE_TYPES = [
    'AndExpression',
    'Sizeof',
    'Identifier',
    'ForInit',
    'ReturnStatement',
    'SizeofOperand',
    'InclusiveOrExpression',
    'PtrMemberAccess',
    'AssignmentExpression',
    'ParameterList',
    'IdentifierDeclType',
    'SizeofExpression',
    'SwitchStatement',
    'IncDec',
    'Function',
    'BitAndExpression',
    'UnaryExpression',
    'DoStatement',
    'GotoStatement',
    'Callee',
    'OrExpression',
    'ShiftExpression',
    'Decl',
    'CFGErrorNode',
    'WhileStatement',
    'InfiniteForNode',
    'RelationalExpression',
    'CFGExitNode',
    'Condition',
    'BreakStatement',
    'CompoundStatement',
    'UnaryOperator',
    'CallExpression',
    'CastExpression',
    'ConditionalExpression',
    'ArrayIndexing',
    'PostIncDecOperationExpression',
    'Label',
    'ArgumentList',
    'EqualityExpression',
    'ReturnType',
    'Parameter',
    'Argument',
    'Symbol',
    'ParameterType',
    'Statement',
    'AdditiveExpression',
    'PrimaryExpression',
    'DeclStmt',
    'CastTarget',
    'IdentifierDeclStatement',
    'IdentifierDecl',
    'CFGEntryNode',
    'TryStatement',
    'Expression',
    'ExclusiveOrExpression',
    'ClassDef',
    'File',
    'UnaryOperationExpression',
    'ClassDefStatement',
    'FunctionDef',
    'IfStatement',
    'MultiplicativeExpression',
    'ContinueStatement',
    'MemberAccess',
    'ExpressionStatement',
    'ForStatement',
    'InitializerList',
    'ElseStatement',
    # Newly added
    'ThrowStatement',
    'CFGExceptionNode'
]
NODE_TYPES_TO_IDS = {
    'AndExpression': 1,
    'Sizeof': 2,
    'Identifier': 3,
    'ForInit': 4,
    'ReturnStatement': 5,
    'SizeofOperand': 6,
    'InclusiveOrExpression': 7,
    'PtrMemberAccess': 8,
    'AssignmentExpression': 9,
    'ParameterList': 10,
    'IdentifierDeclType': 11,
    'SizeofExpression': 12,
    'SwitchStatement': 13,
    'IncDec': 14,
    'Function': 15,
    'BitAndExpression': 16,
    'UnaryExpression': 17,
    'DoStatement': 18,
    'GotoStatement': 19,
    'Callee': 20,
    'OrExpression': 21,
    'ShiftExpression': 22,
    'Decl': 23,
    'CFGErrorNode': 24,
    'WhileStatement': 25,
    'InfiniteForNode': 26,
    'RelationalExpression': 27,
    'CFGExitNode': 28,
    'Condition': 29,
    'BreakStatement': 30,
    'CompoundStatement': 31,
    'UnaryOperator': 32,
    'CallExpression': 33,
    'CastExpression': 34,
    'ConditionalExpression': 35,
    'ArrayIndexing': 36,
    'PostIncDecOperationExpression': 37,
    'Label': 38,
    'ArgumentList': 39,
    'EqualityExpression': 40,
    'ReturnType': 41,
    'Parameter': 42,
    'Argument': 43,
    'Symbol': 44,
    'ParameterType': 45,
    'Statement': 46,
    'AdditiveExpression': 47,
    'PrimaryExpression': 48,
    'DeclStmt': 49,
    'CastTarget': 50,
    'IdentifierDeclStatement': 51,
    'IdentifierDecl': 52,
    'CFGEntryNode': 53,
    'TryStatement': 54,
    'Expression': 55,
    'ExclusiveOrExpression': 56,
    'ClassDef': 57,
    'File': 58,
    'UnaryOperationExpression': 59,
    'ClassDefStatement': 60,
    'FunctionDef': 61,
    'IfStatement': 62,
    'MultiplicativeExpression': 63,
    'ContinueStatement': 64,
    'MemberAccess': 65,
    'ExpressionStatement': 66,
    'ForStatement': 67,
    'InitializerList': 68,
    'ElseStatement': 69,
    # Newly added
    'ThrowStatement': 70,
    'CFGExceptionNode': 71
}
EDGE_TYPES = [
    'IS_AST_PARENT',
    'IS_CLASS_OF',
    'FLOWS_TO',
    'EDF',
    'USE',
    'REACHES',
    'CONTROLS',
    'DECLARES',
    'DOM',
    'POST_DOM',
    'IS_FUNCTION_OF_AST',
    'IS_FUNCTION_OF_CFG'
]
EDGE_TYPES_TO_IDS = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}

TIMEOUT = 3600


def dataframe_to_code(dataframe, output_dir, code_tag='code'):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    for idx, row in dataframe.iterrows():
        with open(osp.join(output_dir, f"{idx}.c"), 'w') as f:
            f.write(row[code_tag])


def analyze_code(code_dir, output_dir):
    def terminate(pid):
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, os.WNOHANG)
        shutil.rmtree('tmp/')

    joern_parse = osp.join(JOERN_DIR, 'joern-parse')
    command = f"{joern_parse} {code_dir} tmp"
    start_date = datetime.datetime.now()
    process = subprocess.Popen(command, shell=True)
    try:
        while process.poll() is None:
            time.sleep(0.5)
            end_date = datetime.datetime.now()
            if (end_date - start_date).seconds > TIMEOUT:
                terminate(process.pid)
    except BaseException as e:
        terminate(process.pid)
        raise e
    shutil.copytree(f'tmp/{code_dir}', output_dir)
    shutil.rmtree('tmp/')


def parse_graph(w2v_model, node_csv, edge_csv):
    with open(node_csv, 'r') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')

        nodes_to_ids, node_features = {}, {}
        for i, node in enumerate(nodes):
            is_cfg_node = node['isCFGNode'].strip()
            if is_cfg_node != 'True':
                continue

            node_key, node_type = node['key'], node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()
            node_split = code_tokenize(node_content)

            nrp = np.zeros(w2v_model.vector_size)
            for token in node_split:
                embedding = w2v_model.wv[token] if token in w2v_model.wv else np.zeros(
                    w2v_model.vector_size)
                nrp += embedding
            f_nrp = np.divide(nrp, len(node_split)) if len(node_split) > 0 else nrp
            type_onehot = np.eye(len(NODE_TYPES_TO_IDS))[NODE_TYPES_TO_IDS[node_type] - 1]
            node_feature = np.concatenate([type_onehot, f_nrp], axis=0)

            node_features[node_key] = node_feature
            nodes_to_ids[node_key] = i

    edges = []
    nodes_on_edge = set()
    with open(edge_csv, 'r') as ec:
        raw_edges = csv.DictReader(ec, delimiter='\t')
        for edge in raw_edges:
            start, end, edge_type = edge["start"], edge["end"], edge["type"]
            if edge_type == "IS_FILE_OF" or edge_type not in EDGE_TYPES_TO_IDS:
                continue
            if start not in nodes_to_ids or end not in nodes_to_ids:
                continue
            edges.append([start, EDGE_TYPES_TO_IDS[edge_type], end])
            nodes_on_edge.update({start, end})

    noes_to_ids = {node: i for i, node in enumerate(nodes_on_edge)}
    nodes = [node_features[node_key] for node_key in nodes_on_edge]
    edges = [[noes_to_ids[start], edge_type_id, noes_to_ids[end]]
             for start, edge_type_id, end in edges]

    try:
        graph = {'node_features': np.stack(nodes, axis=0).tolist(), 'graph': edges}
    except ValueError:
        graph = {'node_features': [], 'graph': edges}
    return graph


def generate_graphs(w2v_model, csv_dir, raw_data, label_field='label'):
    graphs = []
    for i, entry in enumerate(raw_data):
        filename = f'{i}.c'
        node_csv = osp.join(csv_dir, filename, 'nodes.csv')
        edge_csv = osp.join(csv_dir, filename, 'edges.csv')
        label = int(entry[label_field])
        if not osp.exists(node_csv) or not osp.exists(edge_csv):
            continue
        graph = parse_graph(w2v_model, node_csv, edge_csv)
        graph.update({'index': i, 'target': label})
        graphs.append(graph)
    return graphs


def json_to_graphs(w2v_model, json_path, output_dir, code_key='code', label_key='label'):
    # Read
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        dataframe = pd.DataFrame.from_records(raw_data)
    # Process
    dataframe_to_code(dataframe, 'raw_code/', code_key)
    csv_dir = osp.join(output_dir, 'intermediate')
    try:
        analyze_code('raw_code/', csv_dir)
        graphs = generate_graphs(w2v_model, csv_dir, raw_data, label_key)
    finally:
        shutil.rmtree('raw_code/')
        shutil.rmtree(csv_dir)
    return graphs


def csv_to_graphs(w2v_model, csv_path, output_dir, code_field='code', delimiter=','):
    # Read
    with open(csv_path, 'r') as f:
        csv_reader = csv.DictReader(f, delimiter=delimiter)
        raw_data = [row for row in csv_reader]
        dataframe = pd.DataFrame.from_records(raw_data)
    # Process
    dataframe_to_code(dataframe, 'raw_code/', code_field)
    csv_dir = osp.join(output_dir, 'intermediate')
    try:
        analyze_code('raw_code/', csv_dir)
    finally:
        shutil.rmtree('raw_code/')
    graphs = generate_graphs(w2v_model, csv_dir, raw_data)
    return graphs
