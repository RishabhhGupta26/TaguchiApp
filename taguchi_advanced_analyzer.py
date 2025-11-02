# taguchi_advanced_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from scipy import stats

st.set_page_config(layout="wide", page_title="Taguchi’s Orthogonal Array Advanced Analyzer")
st.title("Taguchi’s Orthogonal Array — Advanced Analyzer")

# -------------------------
# Built-in OA (small set)
# -------------------------
def oa_builtin(name):
    if name == "L9 (3-level * 4-factor)":
        data = [
            [1,1,1,1],[1,2,2,2],[1,3,3,3],
            [2,1,2,3],[2,2,3,1],[2,3,1,2],
            [3,1,3,2],[3,2,1,3],[3,3,2,1]
        ]
        return pd.DataFrame(data, columns=["C1","C2","C3","C4"])
    if name == "L8(2-level * 7-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 2, 2, 1, 1],
            [2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 2, 1, 2, 1],
            [2, 2, 1, 1, 2, 2, 1],
            [2, 2, 1, 2, 1, 1, 2]
        ]
        return pd.DataFrame(data, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7"])
    if name == "L8(2-level * 4-factor, 4-level * 1-factor)":
        data = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2],
            [2, 1, 1, 2, 2],
            [2, 2, 2, 1, 1],
            [3, 1, 2, 1, 2],
            [3, 2, 1, 2, 1],
            [4, 1, 2, 2, 1],
            [4, 2, 1, 1, 2]
       ]
        return pd.DataFrame(data, columns=["C1", "C2", "C3", "C4", "C5"])
    if name == "L12(2-level * 11-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            [1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2],
            [1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1],
            [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1],
            [2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2],
            [2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2],
            [2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2],
            [2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1]
       ]
        return pd.DataFrame(data, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"])
    if name == "L16(2-level * 15-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
            [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1],
            [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
            [1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],
            [2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2],
            [2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1]
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 16)])
    if name == "L16(2-level * 12-factor, 4-level * 1-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
            [2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            [2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
            [3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1],
            [3, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],
            [3, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2],
            [4, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [4, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2],
            [4, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2],
            [4, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1]
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 14)])
    if name == "L16(2-level * 9-factor, 4-level * 2-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [1, 3, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            [1, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1],
            [2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2],
            [2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1],
            [2, 3, 2, 1, 1, 1, 2, 2, 2, 1, 1],
            [2, 4, 2, 1, 1, 2, 1, 1, 1, 2, 2],
            [3, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2],
            [3, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1],
            [3, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [3, 4, 1, 2, 1, 1, 2, 1, 2, 1, 2],
            [4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1],
            [4, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2],
            [4, 3, 1, 1, 2, 2, 2, 1, 1, 1, 2],
            [4, 4, 1, 1, 2, 1, 1, 2, 2, 2, 1]
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 12)])
    if name == "L16(4-level * 4-factor, 2-level * 3-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 1, 2, 2, 2],
            [1, 3, 3, 2, 3, 1, 2],
            [1, 4, 4, 2, 4, 2, 1],
            [2, 1, 2, 2, 1, 2, 1],
            [2, 2, 1, 2, 2, 1, 2],
            [2, 3, 4, 1, 3, 2, 2],
            [2, 4, 3, 1, 4, 1, 1],
            [3, 1, 3, 1, 4, 2, 2],
            [3, 2, 4, 1, 3, 1, 1],
            [3, 3, 1, 2, 2, 2, 1],
            [3, 4, 2, 2, 1, 1, 2],
            [4, 1, 4, 2, 2, 1, 2],
            [4, 2, 3, 2, 1, 2, 1],
            [4, 3, 2, 1, 4, 1, 1],
            [4, 4, 1, 1, 3, 2, 2]
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 8)])
    if name == "L16(4-level * 5-factor)":
        data = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2],
            [1, 3, 3, 3, 3],
            [1, 4, 4, 4, 4],
            [2, 1, 2, 3, 4],
            [2, 2, 1, 4, 3],
            [2, 3, 4, 1, 2],
            [2, 4, 3, 2, 1],
            [3, 1, 3, 4, 2],
            [3, 2, 4, 3, 1],
            [3, 3, 1, 2, 4],
            [3, 4, 2, 1, 3],
            [4, 1, 4, 2, 3],
            [4, 2, 3, 1, 4],
            [4, 3, 2, 4, 1],
            [4, 4, 1, 3, 2]
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 6)])
    if name == "L16(8-level * 1-factor, 2-level * 8-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 1, 1, 1, 1],
            [3, 1, 1, 2, 2, 1, 1, 2, 2],
            [3, 2, 2, 1, 1, 2, 2, 1, 1],
            [4, 1, 1, 2, 2, 2, 2, 1, 1],
            [4, 2, 2, 1, 1, 1, 1, 2, 2],
            [5, 1, 2, 1, 2, 1, 2, 1, 2],
            [5, 2, 1, 2, 1, 2, 1, 2, 1],
            [6, 1, 2, 1, 2, 2, 1, 2, 1],
            [6, 2, 1, 2, 1, 1, 2, 1, 2],
            [7, 1, 2, 2, 1, 1, 2, 2, 1],
            [7, 2, 1, 1, 2, 2, 1, 1, 2],
            [8, 1, 2, 2, 1, 2, 1, 1, 2],
            [8, 2, 1, 1, 2, 1, 2, 2, 1],
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 10)])
    if name == "L16(2-level * 6-factor, 4-level * 3-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 1, 1, 2, 2, 2, 2],
            [1, 3, 3, 2, 2, 1, 1, 2, 2],
            [1, 4, 4, 2, 2, 2, 2, 1, 1],
            [2, 1, 2, 2, 2, 1, 2, 1, 2],
            [2, 2, 1, 2, 2, 2, 1, 2, 1],
            [2, 3, 4, 1, 1, 1, 2, 2, 1],
            [2, 4, 3, 1, 1, 2, 1, 1, 2],
            [3, 1, 3, 1, 2, 2, 2, 2, 1],
            [3, 2, 4, 1, 2, 1, 1, 1, 2],
            [3, 3, 1, 2, 1, 2, 2, 1, 2],
            [3, 4, 2, 2, 1, 1, 1, 2, 1],
            [4, 1, 4, 2, 1, 2, 1, 2, 2],
            [4, 2, 3, 2, 1, 1, 2, 1, 1],
            [4, 3, 2, 1, 2, 2, 1, 1, 1],
            [4, 4, 1, 1, 2, 1, 2, 2, 2],
       ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 10)])
    if name == "L18 (2-level * 1-factor, 3-level * 7-factor)":
        data = [
            [1,1,1,1,1,1,1,1],
            [1,1,2,2,2,2,2,2],
            [1,1,3,3,3,3,3,3],
            [1,2,1,2,3,3,3,1],
            [1,2,2,3,1,1,1,2],
            [1,2,3,1,2,2,2,3],
            [1,3,1,3,2,1,2,2],
            [1,3,2,1,3,2,3,3],
            [1,3,3,2,1,3,1,1],
            [2,1,1,3,3,2,2,1],
            [2,1,2,1,1,3,3,2],
            [2,1,3,2,2,1,1,3],
            [2,2,1,1,2,3,1,3],
            [2,2,2,2,3,1,2,1],
            [2,2,3,3,1,2,3,2],
            [2,3,1,2,1,1,3,2],
            [2,3,2,3,2,2,1,3],
            [2,3,3,1,3,3,2,1],
        ]
        return pd.DataFrame(data, columns=["C1","C2","C3","C4","C5","C6","C7","C8"])
    if name == "L18 (6-level * 1-factor, 3-level * 6-factor)":
        data = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2],
            [1, 3, 3, 3, 3, 3, 3],
            [2, 1, 1, 2, 2, 3, 3],
            [2, 2, 2, 3, 3, 1, 1],
            [2, 3, 3, 1, 1, 2, 2],
            [3, 1, 2, 1, 3, 2, 3],
            [3, 2, 3, 2, 1, 3, 1],
            [3, 3, 1, 3, 2, 1, 2],
            [4, 1, 3, 3, 2, 2, 1],
            [4, 2, 1, 1, 3, 3, 2],
            [4, 3, 2, 2, 1, 1, 3],
            [5, 1, 2, 3, 1, 3, 2],
            [5, 2, 3, 1, 2, 1, 3],
            [5, 3, 1, 2, 3, 2, 1],
            [6, 1, 3, 2, 3, 1, 2],
            [6, 2, 1, 3, 1, 2, 3],
            [6, 3, 2, 1, 2, 3, 1],
        ]
        return pd.DataFrame(data, columns=["C1","C2","C3","C4","C5","C6","C7"])
    if name == "L25(5-level * 6-factor)":
        data = [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2],
            [1, 3, 3, 3, 3, 3],
            [1, 4, 4, 4, 4, 4],
            [1, 5, 5, 5, 5, 5],
            [2, 1, 2, 3, 4, 5],
            [2, 2, 3, 4, 5, 1],
            [2, 3, 4, 5, 1, 2],
            [2, 4, 5, 1, 2, 3],
            [2, 5, 1, 2, 3, 4],
            [3, 1, 3, 5, 2, 4],
            [3, 2, 4, 1, 3, 5],
            [3, 3, 5, 2, 4, 1],
            [3, 4, 1, 3, 5, 2],
            [3, 5, 2, 4, 1, 3],
            [4, 1, 4, 2, 5, 3],
            [4, 2, 5, 3, 1, 4],
            [4, 3, 1, 4, 2, 5],
            [4, 4, 2, 5, 3, 1],
            [4, 5, 3, 1, 4, 2],
            [5, 1, 5, 4, 3, 2],
            [5, 2, 1, 5, 4, 3],
            [5, 3, 2, 1, 5, 4],
            [5, 4, 3, 2, 1, 5],
            [5, 5, 4, 3, 2, 1],
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 7)])
    if name == "L27(3-level * 13-factor)":
        data = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,3,3,3,3,3,3,3,3,3],
            [1,2,2,2,1,1,1,2,2,2,3,3,3],
            [1,2,2,2,2,2,2,3,3,3,1,1,1],
            [1,2,2,2,3,3,3,1,1,1,2,2,2],
            [1,3,3,3,1,1,1,3,3,3,2,2,2],
            [1,3,3,3,2,2,2,1,1,1,3,3,3],
            [1,3,3,3,3,3,3,2,2,2,1,1,1],
            [2,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,2,3,2,3,1,2,3,1,2,3,1],
            [2,1,2,3,3,1,2,3,1,2,3,1,2],
            [2,2,3,1,1,2,3,2,3,1,3,1,2],
            [2,2,3,1,2,3,1,3,1,2,1,2,3],
            [2,2,3,1,3,1,2,1,2,3,2,3,1],
            [2,3,1,2,1,2,3,3,1,2,2,3,1],
            [2,3,1,2,2,3,1,1,2,3,3,1,2],
            [2,3,1,2,3,1,2,2,3,1,1,2,3],
            [3,1,3,2,1,3,2,1,3,2,1,3,2],
            [3,1,3,2,2,1,3,2,1,3,2,1,3],
            [3,1,3,2,3,2,1,3,2,1,3,2,1],
            [3,2,1,3,1,3,2,2,1,3,3,2,1],
            [3,2,1,3,2,1,3,3,2,1,1,3,2],
            [3,2,1,3,3,2,1,1,3,2,2,1,3],
            [3,3,2,1,1,3,2,3,2,1,2,1,3],
            [3,3,2,1,2,1,3,1,3,2,3,2,1],
            [3,3,2,1,3,2,1,2,1,3,1,3,2]
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 14)])
    if name == "L32 (2-level * 31-factor)":
        data = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,2,2,2,2,2,1,1,1,1,2,2,2,2],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,2,1,1,1,1],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,1,1,1,1],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 32)])
    if name == "L32 (2-level * 1-factor, 4-level * 9-factor)":
        data = [
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,2,2,2,2,2,2,2,2],
            [1,1,3,3,3,3,3,3,3,3],
            [1,1,4,4,4,4,4,4,4,4],
            [1,2,1,1,2,2,3,3,4,4],
            [1,2,2,2,1,1,4,4,3,3],
            [1,2,3,3,4,4,1,1,2,2],
            [1,2,4,4,3,3,2,2,1,1],
            [1,3,1,2,3,4,1,2,3,4],
            [1,3,2,1,4,3,2,1,4,3],
            [1,3,3,4,1,2,3,4,1,2],
            [1,3,4,3,2,1,4,3,2,1],
            [1,4,1,2,4,3,3,4,2,1],
            [1,4,2,1,3,4,4,3,1,2],
            [1,4,3,4,2,1,1,2,4,3],
            [1,4,4,3,1,2,2,1,3,4],
            [2,1,1,4,1,4,2,3,2,3],
            [2,1,2,3,2,3,1,4,1,4],
            [2,1,3,2,3,2,4,1,4,1],
            [2,1,4,1,4,1,3,2,3,2],
            [2,2,1,4,2,3,4,1,3,2],
            [2,2,2,3,1,4,3,2,4,1],
            [2,2,3,2,4,1,2,3,1,4],
            [2,2,4,1,3,2,1,4,2,3],
            [2,3,1,3,3,1,2,4,4,2],
            [2,3,2,4,4,2,1,3,3,1],
            [2,3,3,1,1,3,4,2,2,4],
            [2,3,4,2,2,4,3,1,1,3],
            [2,4,1,3,4,2,4,2,1,3],
            [2,4,2,4,3,1,3,1,2,4],
            [2,4,3,1,2,4,2,4,3,1],
            [2,4,4,2,1,3,1,3,4,2]
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 11)])
    if name == "L36 (2-level * 11-factor, 3-level * 12-factor)":
        data = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,3,3,3,3],
            [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,1,1,1,1],
            [1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,1,1,1,1,2,2,2,2],
            [1,1,2,2,2,1,1,1,2,2,2,1,1,2,3,1,2,3,3,1,2,2,3],
            [1,1,2,2,2,1,1,1,2,2,2,2,2,3,1,2,3,1,1,2,3,3,1],
            [1,1,2,2,2,1,1,1,2,2,2,3,3,1,2,3,1,2,2,3,1,1,2],
            [1,2,1,2,2,1,2,2,1,1,2,1,1,3,2,1,3,2,3,2,1,3,2],
            [1,2,1,2,2,1,2,2,1,1,2,2,2,1,3,2,1,3,1,3,2,1,3],
            [1,2,1,2,2,1,2,2,1,1,2,3,3,2,1,3,2,1,2,1,3,2,1],
            [1,2,2,1,2,2,1,2,1,2,1,1,2,3,1,3,2,1,3,3,2,1,2],
            [1,2,2,1,2,2,1,2,1,2,1,2,3,1,2,1,3,2,1,1,3,2,3],
            [1,2,2,1,2,2,1,2,1,2,1,3,1,2,3,2,1,3,2,2,1,3,1],
            [1,2,2,2,1,2,2,1,2,1,1,1,2,3,2,1,1,3,2,3,3,2,1],
            [1,2,2,2,1,2,2,1,2,1,1,2,3,1,3,2,2,1,3,1,1,3,2],
            [1,2,2,2,1,2,2,1,2,1,1,3,1,2,1,3,3,2,1,2,2,1,3],
            [2,1,2,2,1,1,2,2,1,2,1,1,2,1,3,3,3,1,2,2,1,2,3],
            [2,1,2,2,1,1,2,2,1,2,1,2,3,2,1,1,1,2,3,3,2,3,1],
            [2,1,2,2,1,1,2,2,1,2,1,3,1,3,2,2,2,3,1,1,3,1,2],
            [2,1,2,1,2,2,2,1,1,1,2,1,2,2,3,3,1,2,1,1,3,3,2],
            [2,1,2,1,2,2,2,1,1,1,2,2,3,3,1,1,2,3,2,2,1,1,3],
            [2,1,2,1,2,2,2,1,1,1,2,3,1,1,2,2,3,1,3,3,2,2,1],
            [2,1,1,2,2,2,1,2,2,1,1,1,3,2,1,2,3,3,1,3,1,2,2],
            [2,1,1,2,2,2,1,2,2,1,1,2,1,3,2,3,1,1,2,1,2,3,3],
            [2,1,1,2,2,2,1,2,2,1,1,3,2,1,3,1,2,2,3,2,3,1,1],
            [2,2,2,1,1,1,1,2,2,1,2,1,3,2,2,2,1,1,3,2,3,1,3],
            [2,2,2,1,1,1,1,2,2,1,2,2,1,3,3,3,2,2,1,3,1,2,1],
            [2,2,2,1,1,1,1,2,2,1,2,3,2,1,1,1,3,3,2,1,2,3,2],
            [2,2,1,2,1,2,1,1,1,2,2,1,3,3,3,2,3,2,2,1,2,1,1],
            [2,2,1,2,1,2,1,1,1,2,2,2,1,1,1,3,1,3,3,2,3,2,2],
            [2,2,1,2,1,2,1,1,1,2,2,3,2,2,2,1,2,1,1,3,1,3,3],
            [2,2,1,1,2,1,2,1,2,2,1,1,3,1,2,3,2,3,1,2,2,3,1],
            [2,2,1,1,2,1,2,1,2,2,1,2,1,2,3,1,3,1,2,3,3,1,2],
            [2,2,1,1,2,1,2,1,2,2,1,3,2,3,1,2,1,2,3,1,2,3,2]
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 24)])
    if name == "L36 (2-level * 3-factor, 3-level * 13-factor)":
        data = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,2,2,1,1,1,1,1,2,2,2,2,3,3,3,3],
            [1,2,2,1,2,2,2,2,3,3,3,3,1,1,1,1],
            [1,2,2,1,3,3,3,3,1,1,1,1,2,2,2,2],
            [2,1,2,1,1,1,2,3,1,2,3,3,1,2,2,3],
            [2,1,2,1,2,2,3,1,2,3,1,1,2,3,3,1],
            [2,1,2,1,3,3,1,2,3,1,2,2,3,1,1,2],
            [2,2,1,1,1,1,3,2,1,3,2,3,2,1,3,2],
            [2,2,1,1,2,2,1,3,2,1,3,1,3,2,1,3],
            [2,2,1,1,3,3,2,1,3,2,1,2,1,3,2,1],
            [1,1,1,2,1,2,3,1,3,2,1,3,3,2,1,2],
            [1,1,1,2,2,3,1,2,1,3,2,1,1,3,2,3],
            [1,1,1,2,3,1,2,3,2,1,3,2,2,1,3,1],
            [1,2,2,2,1,2,3,2,1,1,3,2,3,3,2,1],
            [1,2,2,2,2,3,1,3,2,2,1,3,1,1,3,2],
            [1,2,2,2,3,1,2,1,3,3,2,1,2,2,1,3],
            [2,1,2,2,1,2,1,3,3,3,1,2,2,1,2,3],
            [2,1,2,2,2,3,2,1,1,1,2,3,3,2,3,1],
            [2,1,2,2,3,1,3,2,2,2,3,1,1,3,1,2],
            [2,2,1,2,1,2,2,3,3,1,2,1,1,3,3,2],
            [2,2,1,2,2,3,3,1,1,2,3,2,2,1,1,3],
            [2,2,1,2,3,1,1,2,2,3,1,3,3,2,2,1],
            [1,1,1,3,1,3,2,1,2,3,3,1,3,1,2,2],
            [1,1,1,3,2,1,3,2,3,1,1,2,1,2,3,3],
            [1,1,1,3,3,2,1,3,1,2,2,3,2,3,1,1],
            [1,2,2,3,1,3,2,2,2,1,1,3,2,3,1,3],
            [1,2,2,3,2,1,3,3,3,2,2,1,3,1,2,1],
            [1,2,2,3,3,2,1,1,1,3,3,2,1,2,3,2],
            [2,1,2,3,1,3,3,3,2,3,2,2,1,2,1,1],
            [2,1,2,3,2,1,1,1,3,1,3,3,2,3,2,2],
            [2,1,2,3,3,2,2,2,1,2,1,1,3,1,3,3],
            [2,2,1,3,1,3,1,2,3,2,3,1,2,2,3,1],
            [2,2,1,3,2,1,2,3,1,3,1,2,3,3,1,2],
            [2,2,1,3,3,2,3,1,2,1,2,3,1,1,2,3]
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 17)])
    if name == "L54 (2-level * 1-factor, 3-level * 25-factor)":
        data = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,2,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3],
            [1,2,1,1,2,2,3,3,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1],
            [1,2,1,1,2,2,3,3,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2],
            [1,2,2,2,3,3,1,1,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3],
            [1,2,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1],
            [1,2,2,2,3,3,1,1,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2],
            [1,2,3,3,1,1,2,2,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3],
            [1,2,3,3,1,1,2,2,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1],
            [1,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2],
            [1,3,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3],
            [1,3,1,2,1,3,2,3,2,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,3,1],
            [1,3,1,2,1,3,2,3,3,1,3,2,1,2,3,1,3,2,1,2,3,1,3,2,1,2],
            [1,3,2,3,2,1,3,1,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3],
            [1,3,2,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,3,1],
            [1,3,2,3,2,1,3,1,3,1,3,2,1,2,3,1,3,2,1,2,3,1,3,2,1,2],
            [1,3,3,1,3,2,1,2,1,2,1,3,2,3,1,2,1,3,2,3,1,2,1,3,2,3],
            [1,3,3,1,3,2,1,2,2,3,2,1,3,1,2,3,2,1,3,1,2,3,2,1,3,1],
            [1,3,3,1,3,2,1,2,3,1,3,2,1,2,3,1,3,2,1,2,3,1,3,2,1,2],
            [2,1,1,3,3,2,2,1,1,3,3,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,1,3,3,2,2,1,2,1,1,3,3,1,2,3,1,2,3,1,2,3,1,2,3,1],
            [2,1,1,3,3,2,2,1,3,2,2,1,1,2,3,1,2,3,1,2,3,1,2,3,1,2],
            [2,1,2,1,1,3,3,2,1,3,3,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,2,1,1,3,3,2,2,1,1,3,3,1,2,3,1,2,3,1,2,3,1,2,3,1],
            [2,1,2,1,1,3,3,2,3,2,2,1,1,2,3,1,2,3,1,2,3,1,2,3,1,2],
            [2,1,3,2,2,1,1,3,1,3,3,2,2,3,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,3,2,2,1,1,3,2,1,1,3,3,1,2,3,1,2,3,1,2,3,1,2,3,1],
            [2,1,3,2,2,1,1,3,3,2,2,1,1,2,3,1,2,3,1,2,3,1,2,3,1,2],
            [2,2,1,2,3,1,3,2,1,2,3,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2],
            [2,2,1,2,3,1,3,2,2,3,1,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3],
            [2,2,1,2,3,1,3,2,3,1,2,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1],
            [2,2,2,3,1,2,1,3,1,2,3,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2],
            [2,2,2,3,1,2,1,3,2,3,1,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3],
            [2,2,2,3,1,2,1,3,3,1,2,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1],
            [2,2,3,1,2,3,2,1,1,2,3,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2],
            [2,2,3,1,2,3,2,1,2,3,1,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3],
            [2,2,3,1,2,3,2,1,3,1,2,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1],
            [2,3,1,3,2,3,1,2,1,3,2,3,1,3,2,3,1,3,2,3,1,3,2,3,1,3],
            [2,3,1,3,2,3,1,2,2,1,3,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1],
            [2,3,1,3,2,3,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2],
            [2,3,2,1,3,1,2,3,1,3,2,3,1,3,2,3,1,3,2,3,1,3,2,3,1,3],
            [2,3,2,1,3,1,2,3,2,1,3,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1],
            [2,3,2,1,3,1,2,3,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2],
            [2,3,3,2,1,2,3,1,1,3,2,3,1,3,2,3,1,3,2,3,1,3,2,3,1,3],
            [2,3,3,2,1,2,3,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1],
            [2,3,3,2,1,2,3,1,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2]
        ]
        return pd.DataFrame(data, columns=[f"C{i}" for i in range(1, 27)])
    if name == "L4 (2-level * 3-factor)":
        return pd.DataFrame([[1,1,1],[1,2,2],[2,1,2],[2,2,1]], columns=["C1","C2","C3"])
    return pd.DataFrame()

# -------------------------
# S/N functions
# -------------------------
def sn_smaller(y):
    y = np.array(y, dtype=float)
    if y.size == 0:
        return np.nan
    mse = np.mean(y**2)
    mse = max(mse, 1e-12)
    return -10.0 * np.log10(mse)

def sn_larger(y):
    y = np.array(y, dtype=float)
    if y.size == 0:
        return np.nan
    inv_mean = np.mean(1.0 / (y**2 + 1e-12))
    inv_mean = max(inv_mean, 1e-12)
    return -10.0 * np.log10(inv_mean)

def sn_nominal(y, target):
    y = np.array(y, dtype=float)
    if y.size == 0:
        return np.nan
    sigma2 = np.var(y, ddof=0)
    sigma2 = max(sigma2, 1e-12)
    mean_y = np.mean(y)
    # Use ratio form (dB)
    return 10 * np.log10((mean_y ** 2) / sigma2)

# -------------------------
# UI: data source
# -------------------------
left, right = st.columns([1,2])
with left:
    mode = st.radio("Data source", ["Built-in OA", "Upload experiment file (CSV/XLSX)"])
    design = None
    if mode == "Built-in OA":
        builtin_choice = st.selectbox("Choose built-in OA", ["L4 (2-level * 3-factor)", "L8(2-level * 4-factor, 4-level * 1-factor)","L8(2-level * 7-factor)", "L9 (3-level * 4-factor)", "L12(2-level * 11-factor)", "L16(2-level * 15-factor)", "L16(2-level * 12-factor, 4-level * 1-factor)", "L16(2-level * 9-factor, 4-level * 2-factor)", "L16(4-level * 4-factor, 2-level * 3-factor)", "L16(4-level * 5-factor)", "L16(8-level * 1-factor, 2-level * 8-factor)", "L16(2-level * 6-factor, 4-level * 3-factor)", "L18 (2-level * 1-factor, 3-level * 7-factor)", "L18 (6-level * 1-factor, 3-level * 6-factor)", "L25(5-level * 6-factor)", "L27(3-level * 13-factor)", "L32 (2-level * 31-factor)", "L32 (2-level * 1-factor, 4-level * 9-factor)", "L36 (2-level * 11-factor, 3-level * 12-factor)", "L36 (2-level * 3-factor, 3-level * 13-factor)", "L54 (2-level * 1-factor, 3-level * 25-factor)"])
        design = oa_builtin(builtin_choice)
        st.write("Design (coded levels):")
        st.dataframe(design)
    else:
        uploaded = st.file_uploader("Upload CSV/XLSX (single sheet or multi-sheet Excel)", type=["csv","xlsx"]) 
        design = None

with right:
    st.markdown("### S/N options")
    sn_type = st.selectbox("S/N type", ["Smaller-the-better","Larger-the-better","Nominal-the-best"])
    tgt = None
    if sn_type == "Nominal-the-best":
        tgt = st.number_input("Target value (Nominal-the-best)", value=0.0, step=0.1)
    multi_response = st.checkbox("Allow multiple response columns", value=True)

# -------------------------
# Load or build dataframe (robust Excel handling)
# -------------------------
df = None
if mode == "Built-in OA":
    st.info("You can use demo responses or upload a matching response file.")
    use_demo = st.checkbox("Use demo responses (quick)", value=True)
    if use_demo:
        np.random.seed(42)
        df = design.copy()
        df["Response1"] = np.round(np.random.uniform(1.0,10.0,size=len(df)),4)
    else:
        file2 = st.file_uploader("Upload CSV/XLSX with responses", type=["csv","xlsx"], key="builtin_upload")
        if file2 is None:
            st.stop()
        else:
            if file2.name.lower().endswith(".csv"):
                df = pd.read_csv(file2)
            else:
                try:
                    xls = pd.ExcelFile(file2)
                    if len(xls.sheet_names) > 1:
                        sheet_choice = st.selectbox("Select sheet to use", xls.sheet_names, key="builtin_sheet")
                        df = pd.read_excel(xls, sheet_name=sheet_choice)
                    else:
                        df = pd.read_excel(xls)
                    st.success(f"✅ Loaded sheet for built-in: {file2.name}")
                except Exception as e:
                    st.error(f"❌ Excel load failed: {e}")
                    st.stop()
else:
    # uploaded path
    if 'uploaded' not in locals() or uploaded is None:
        st.stop()
    else:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            try:
                xls = pd.ExcelFile(uploaded)
                if len(xls.sheet_names) > 1:
                    sheet_choice = st.selectbox("Select sheet to use", xls.sheet_names, key="upload_sheet")
                    df = pd.read_excel(xls, sheet_name=sheet_choice)
                else:
                    df = pd.read_excel(uploaded)
                st.success(f"✅ Loaded sheet: {uploaded.name}")
            except Exception as e:
                st.error(f"❌ Excel load failed: {e}")
                st.stop()
# --- Clean up invalid numeric columns ---
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
for c in df.columns:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors='ignore')

original_df = df.copy()

# -------------------------
# Factor & Response selection BEFORE experimental data
# -------------------------
st.markdown("### Select factor columns (OA columns) and response column(s)")
possible_cols = list(df.columns)
default_factors = [c for c in possible_cols if c.lower().startswith(("c", "a", "b", "temp", "f"))][:3]
selected_factors = st.multiselect("Factor columns", possible_cols, default=default_factors)
if not selected_factors:
    st.warning("Pick at least one factor column.")
    st.stop()

possible_responses = [c for c in df.columns if c not in selected_factors]
selected_responses = st.multiselect("Response column(s)", possible_responses, default=possible_responses[:1] if possible_responses else [])
if not selected_responses:
    st.warning("Pick at least one response column.")
    st.stop()

# -------------------------
# Add multiple new responses dynamically
# -------------------------
st.markdown("### ➕ Add new response columns")
new_resp_list = st.text_area("Enter new response column names separated by commas", placeholder="e.g. Response2, Response3, Response4")
if new_resp_list:
    names = [x.strip() for x in new_resp_list.split(",") if x.strip()]
    for n in names:
        if n not in df.columns:
            df[n] = np.nan
    st.success(f"Added {len(names)} response columns.")
    # refresh possible lists
    possible_responses = [c for c in df.columns if c not in selected_factors]


# --- Refresh the Response column multiselect dynamically ---
st.markdown("### 🔄 Update and Rename Response Columns")

# Ensure new responses added are included automatically
possible_responses = [c for c in df.columns if c not in selected_factors]

# Preserve previously selected ones if still valid
default_resps = [r for r in possible_responses if r in selected_responses] or possible_responses[:1]

# Give unique key to avoid StreamlitDuplicateElementId error
selected_responses = st.multiselect(
    "Select Response column(s)",
    options=possible_responses,
    default=default_resps,
    key="select_response_unique"
)

# Optional renaming UI for each response (live sync)
if selected_responses:
    rename_resp_dict = {}
    cols_resp = st.columns(len(selected_responses))
    for i, resp in enumerate(selected_responses):
        with cols_resp[i]:
            # include index in key to guarantee uniqueness even if names repeat temporarily
            new_name = st.text_input(f"Rename {resp} →", value=resp, key=f"rename_resp_{i}_{resp}")
            rename_resp_dict[resp] = new_name

    if any(rename_resp_dict[r] != r for r in rename_resp_dict):
        # safely rename without changing the iteration keys unexpectedly
        df.rename(columns=rename_resp_dict, inplace=True)
        # Update lists to reflect rename
        selected_responses = [rename_resp_dict.get(r, r) for r in selected_responses]
        possible_responses = [c for c in df.columns if c not in selected_factors]
        st.success("✅ Response column(s) renamed and synced successfully!")

# -------------------------
# Rename factor column headers (optional) - safe check
# -------------------------
st.markdown("### ✏️ Rename factor column headers (optional)")
if len(selected_factors) == 0:
    st.warning("No factor selected to rename.")
else:
    rename_dict = {}
    cols = st.columns(len(selected_factors))
    for i, fac in enumerate(selected_factors):
        with cols[i]:
            new_name = st.text_input(f"Rename {fac} →", value=fac, key=f"rename_fac_{i}_{fac}")
            rename_dict[fac] = new_name

    # Apply rename globally to dataframe only if any name changed
    if any(rename_dict[f] != f for f in rename_dict):
        df.rename(columns=rename_dict, inplace=True)
        # update selected_factors to new names
        selected_factors = [rename_dict.get(f, f) for f in selected_factors]
        st.success("✅ Column renaming applied to all data sections.")
        # update possible_responses after rename
        possible_responses = [c for c in df.columns if c not in selected_factors]

# -------------------------
# Map coded levels → real-world values (optional)
# -------------------------
st.markdown("### Map coded levels → real-world values (optional)")
st.info("Enter mapping for each factor. The actual experimental data below will update immediately.")

mapping_dict = {}
for fac in selected_factors:
    st.markdown(f"#### Mapping for {fac}")
    # guard: if column missing (rename / user error) skip
    if fac not in df.columns:
        st.warning(f"Factor {fac} not found in data (maybe renamed). Skipping mapping for this factor.")
        continue
    levels = sorted(df[fac].dropna().unique(), key=lambda x: str(x))
    two_cols = st.columns(2)
    fac_map = {}
    for i, lvl in enumerate(levels):
        with two_cols[i % 2]:
            # make key unique using fac, index and level representation
            new_val = st.text_input(f"{fac}: {lvl} →", value=str(lvl), key=f"map_{fac}_{i}_{str(lvl)}")
            try:
                new_val_num = float(new_val)
                fac_map[lvl] = new_val_num
            except Exception:
                fac_map[lvl] = new_val  # leave as string if not numeric
    mapping_dict[fac] = fac_map

# Apply mapping into df_mapped
df_mapped = df.copy()
for fac, mp in mapping_dict.items():
    if fac in df_mapped.columns:
        df_mapped[fac] = df_mapped[fac].map(mp).fillna(df_mapped[fac])

# -------------------------
# Editable experimental data (mapped live) - kept smaller height for perf
# -------------------------
st.subheader("Experimental data (editable, reflects mapping)")
st.info("You can directly edit any factor or response value below. For very large datasets, scrolling may be slow.")
edited = st.data_editor(
    df_mapped,
    num_rows="dynamic",
    use_container_width=True,
    height=400  # prevents lag on big sheets
)
# ensure df is always consistent with edited table
df = edited.copy()
# also ensure df_mapped sync
df_mapped = df.copy()

# -------------------------
# Helper: compute S/N for a selected group
# -------------------------
def compute_sn_group_raw(y_array, sn_type, tgt=None):
    if len(y_array) == 0:
        return np.nan
    if sn_type == "Smaller-the-better":
        return sn_smaller(y_array)
    if sn_type == "Larger-the-better":
        return sn_larger(y_array)
    return sn_nominal(y_array, tgt)

# -------------------------
# Multi-response weights UI (only if more than one response)
# -------------------------
weights = None
if multi_response and len(selected_responses) > 1:
    st.markdown("#### Set weights for responses (will be normalized)")
    w_vals = []
    cols = st.columns(len(selected_responses))
    for i, r in enumerate(selected_responses):
        with cols[i]:
            w = st.number_input(f"Weight for {r}", value=1.0, min_value=0.0, format="%.3f", key=f"w_{i}_{r}")
        w_vals.append(w)
    w_arr = np.array(w_vals, dtype=float)
    if w_arr.sum() == 0:
        w_arr = np.ones_like(w_arr)
    weights = w_arr / w_arr.sum()

# -------------------------
# Analysis: Means & S/N per factor level (on df_mapped)
# -------------------------
results = {resp: {} for resp in selected_responses}
for resp in selected_responses:
    for fac in selected_factors:
        # guard if factor missing
        if fac not in df_mapped.columns:
            results[resp][fac] = pd.DataFrame(columns=["Level","Count","Mean","S/N (dB)"]).set_index("Level")
            continue
        levels = pd.Series(df_mapped[fac].dropna().unique())
        # sort numeric-like levels numerically where possible, else lexicographically
        try:
            levels_sorted = sorted(levels.tolist(), key=lambda x: (float(x) if str(x).replace('.','',1).isdigit() else x))
        except Exception:
            levels_sorted = sorted(levels.tolist(), key=lambda x: str(x))
        rows = []
        for lvl in levels_sorted:
            vals = df_mapped[df_mapped[fac] == lvl][resp].dropna().values
            cnt = len(vals)
            meanv = np.nan if cnt == 0 else np.mean(vals)
            snv = np.nan if cnt == 0 else compute_sn_group_raw(vals, sn_type, tgt)
            rows.append([lvl, cnt, meanv, snv])
        results[resp][fac] = pd.DataFrame(rows, columns=["Level","Count","Mean","S/N (dB)"]).set_index("Level")

# Combined S/N (weighted average across responses) if requested
combined_sn = {}
if multi_response and len(selected_responses) > 1 and weights is not None:
    for fac in selected_factors:
        levs = results[selected_responses[0]][fac].index
        comb = pd.Series(0.0, index=levs, dtype=float)
        for i, resp in enumerate(selected_responses):
            s = results[resp][fac]["S/N (dB)"].reindex(levs)
            comb = comb + weights[i] * s.fillna(0.0)
        combined_sn[fac] = comb
else:
    # default: use primary response's S/N
    primary = selected_responses[0]
    for fac in selected_factors:
        combined_sn[fac] = results[primary][fac]["S/N (dB)"]

# -------------------------
# Display per-factor tables (Means & S/N per level)
# -------------------------
st.header("Per-factor tables (Means & S/N per level)")
for fac in selected_factors:
    st.markdown(f"#### Factor: {fac}")
    cols = st.columns(len(selected_responses))
    for i, resp in enumerate(selected_responses):
        with cols[i]:
            st.write(f"Response: {resp}")
            st.dataframe(results[resp][fac].round(4))

# -------------------------
# Main Effects Plots (Mean & S/N)
# -------------------------
st.header("Main Effects Plots")
st.markdown("**Mean response vs Factor level**")
cols_mean = st.columns(len(selected_factors))
for i, fac in enumerate(selected_factors):
    levs = list(results[selected_responses[0]][fac].index)
    mean_vals = results[selected_responses[0]][fac]["Mean"].values
    fig_mean = go.Figure()
    fig_mean.add_trace(go.Scatter(x=levs, y=mean_vals, mode="lines+markers",
                                  line=dict(width=3), marker=dict(size=8)))
    fig_mean.update_layout(title=f"{fac} (Mean)", xaxis_title="Level",
                           yaxis_title=selected_responses[0],
                           title_x=0.5, title_font=dict(size=14), height=300)
    cols_mean[i].plotly_chart(fig_mean, use_container_width=True)

st.markdown("**Mean S/N vs Factor level**")
cols_sn = st.columns(len(selected_factors))
for i, fac in enumerate(selected_factors):
    levs = list(results[selected_responses[0]][fac].index)
    sn_vals = combined_sn[fac].reindex(levs).values
    fig_sn = go.Figure()
    fig_sn.add_trace(go.Scatter(x=levs, y=sn_vals, mode="lines+markers",
                                line=dict(width=3), marker=dict(size=8)))
    fig_sn.update_layout(title=f"{fac} (S/N dB)", xaxis_title="Level",
                         yaxis_title="S/N (dB)", title_x=0.5, title_font=dict(size=14), height=300)
    cols_sn[i].plotly_chart(fig_sn, use_container_width=True)

# -------------------------
# Percent contribution (SS) – corrected to sum to 100%
# -------------------------
def percent_contrib_table(df_for_calc, factors, response_col):
    y = df_for_calc[response_col].dropna().values
    N = len(y)
    grand = np.mean(y) if N > 0 else 0.0
    total_ss = np.sum((y - grand)**2)
    ss = {}
    for f in factors:
        if f not in df_for_calc.columns:
            ss[f] = 0.0
            continue
        grp = df_for_calc.groupby(f)[response_col].agg(['mean','count'])
        ss_f = np.sum(grp['count'] * (grp['mean'] - grand)**2) if not grp.empty else 0.0
        ss[f] = float(ss_f)
    residual = max(0.0, total_ss - sum(ss.values()))
    ss_dict = {**ss, "Residual": residual}
    ss_df = pd.DataFrame.from_dict(ss_dict, orient="index", columns=["SS"])
    total_val = ss_df["SS"].sum()
    ss_df.loc["Total"] = total_val
    if total_val == 0:
        ss_df["%Contribution"] = 0.0
    else:
        ss_df["%Contribution"] = ss_df["SS"] / total_val * 100.0
    return ss_df, grand, total_ss

# compute for the primary response (used for display)
ss_df, grand_mean, total_ss = percent_contrib_table(df_mapped, selected_factors, selected_responses[0])
st.header("Sum of Squares & % Contribution (corrected)")
st.dataframe(ss_df.round(6))

# -------------------------
# Improved % Contribution Chart
# -------------------------
st.markdown("### 🎨 % Contribution Chart")
ss_plot_df = ss_df.drop(index="Total", errors="ignore").copy()
# exclude residual for stacked clarity if you want only factors:
plot_df = ss_plot_df[ss_plot_df.index != "Residual"].copy()
if plot_df.empty:
    st.info("No contribution data to plot.")
else:
    fig_contrib = go.Figure(
        go.Bar(
            x=plot_df.index,
            y=plot_df["%Contribution"],
            text=[f"{v:.1f}%" for v in plot_df["%Contribution"]],
            textposition="outside",
            marker=dict(
                color=plot_df["%Contribution"],
                colorscale="Viridis",
                line=dict(width=1, color="black")
            )
        )
    )
    fig_contrib.update_layout(
        title="Factor % Contribution",
        xaxis_title="Factors",
        yaxis_title="% Contribution",
        template="plotly_white",
        height=360,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

# -------------------------
# ANOVA full table per factor (one-way ANOVA)
# -------------------------
def anova_oneway_table(df_for_calc, factor, response_col):
    groups = df_for_calc.groupby(factor)[response_col].apply(lambda s: s.dropna().values)
    groups = [g for g in groups if len(g) > 0]
    if not groups:
        return None
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = np.mean(np.concatenate(groups)) if N > 0 else np.nan
    ss_between = 0.0
    for g in groups:
        ss_between += len(g) * (np.mean(g) - grand_mean)**2
    df_between = max(0, k - 1)
    ss_within = 0.0
    for g in groups:
        ss_within += np.sum((g - np.mean(g))**2)
    df_within = max(0, N - k)
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    F = ms_between / ms_within if (not np.isnan(ms_between) and not np.isnan(ms_within) and ms_within != 0) else np.nan
    p = stats.f.sf(F, df_between, df_within) if (not np.isnan(F) and df_between > 0 and df_within > 0) else np.nan
    table = pd.DataFrame({
        "Source": [factor, "Residual", "Total"],
        "DF": [df_between, df_within, df_between + df_within],
        "SS": [ss_between, ss_within, ss_between + ss_within],
        "MS": [ms_between, ms_within, np.nan],
        "F": [F, np.nan, np.nan],
        "p-value": [p, np.nan, np.nan]
    }).set_index("Source")
    return table

anova_tables = {}
for f in selected_factors:
    try:
        tbl = anova_oneway_table(df_mapped, f, selected_responses[0])
    except Exception:
        tbl = None
    anova_tables[f] = tbl

# st.subheader("ANOVA tables (one-way per factor)")
# for f, tbl in anova_tables.items():
#     st.markdown(f"**ANOVA — Factor: {f}**")
#     if tbl is None:
#         st.write("ANOVA could not be computed for this factor (insufficient or invalid data).")
#     else:
#         st.dataframe(tbl.round(6))
# st.subheader("ANOVA tables (one-way per factor)")
# for f, tbl in anova_tables.items():
#     st.markdown(f"**ANOVA — Factor: {f}**")
#     if tbl is None:
#         st.write("ANOVA could not be computed for this factor (insufficient or invalid data).")
#     else:
#         # Create display version with "-" instead of NaN
#         display_tbl = tbl.copy()
#         display_tbl[["MS", "F", "p-value"]] = display_tbl[["MS", "F", "p-value"]].replace({np.nan: "-"})
#         st.dataframe(display_tbl.round(6))


st.subheader("ANOVA tables (one-way per factor)")
for f, tbl in anova_tables.items():
    st.markdown(f"**ANOVA — Factor: {f}**")
    if tbl is None:
        st.write("ANOVA could not be computed for this factor (insufficient or invalid data).")
    else:
        # Create a clean display version with proper data types
        display_tbl = tbl.copy()
        
        # Convert numeric columns to proper numeric types first
        for col in ["MS", "F", "p-value"]:
            if col in display_tbl.columns:
                display_tbl[col] = pd.to_numeric(display_tbl[col], errors='coerce')
        
        # Now display with formatting
        formatted_tbl = display_tbl.copy()
        for col in ["MS", "F", "p-value"]:
            if col in formatted_tbl.columns:
                # Format numbers and replace NaN with "-" for display only
                formatted_tbl[col] = formatted_tbl[col].apply(
                    lambda x: f"{x:.6f}" if pd.notna(x) and x != "" else "-"
                )
        
        st.dataframe(formatted_tbl)        
# -------------------------
# Optimal levels & predictions (supports separate per-response or weighted)
# -------------------------
st.header("Optimal levels & predictions")

# Decide active responses
if multi_response and len(selected_responses) > 1 and weights is not None:
    st.info("Using weighted average across responses for optimal combination (Weighted Avg).")
    active_responses = ["Weighted Avg"]
else:
    st.info("Showing separate optimal combinations for each response.")
    active_responses = selected_responses

# For each active response compute optimal combination
optimal_results = {}
for resp in active_responses:
    st.subheader(f"🎯 Optimal combination for: {resp}")
    if resp == "Weighted Avg":
        # overall baseline: average of S/N across responses
        overall_sn = np.nanmean([compute_sn_group_raw(df_mapped[r].dropna().values, sn_type, tgt) for r in selected_responses])
    else:
        overall_sn = compute_sn_group_raw(df_mapped[resp].dropna().values, sn_type, tgt)

    optimal_mapped = {}
    optimal_coded = {}
    for f in selected_factors:
        # choose S/N series depending on type
        if resp == "Weighted Avg" and multi_response and weights is not None:
            series_sn = combined_sn[f]
        else:
            series_sn = results[resp][f]["S/N (dB)"]
        if series_sn.empty or series_sn.isnull().all():
            opt = np.nan
        else:
            opt = series_sn.idxmax()
        optimal_mapped[f] = opt
        optimal_coded[f] = opt

    # predicted S/N by adding level effects
    level_effects = {f: ( (combined_sn[f] if resp == "Weighted Avg" and weights is not None else results[resp][f]["S/N (dB)"]) - overall_sn ) for f in selected_factors}
    predicted_sn = float(overall_sn) if not np.isnan(overall_sn) else np.nan
    for f in selected_factors:
        opt = optimal_mapped[f]
        if isinstance(level_effects[f], pd.Series) and opt in level_effects[f].index:
            predicted_sn += float(level_effects[f].loc[opt])

    # predicted mean
    ss_df_local, grand_mean_local, _ = percent_contrib_table(df_mapped, selected_factors, resp if resp != "Weighted Avg" else selected_responses[0])
    predicted_mean = float(grand_mean_local) if not np.isnan(grand_mean_local) else np.nan
    for f in selected_factors:
        opt = optimal_mapped[f]
        if opt in results[(selected_responses[0] if resp == "Weighted Avg" else resp)][f].index:
            predicted_mean += float(results[(selected_responses[0] if resp == "Weighted Avg" else resp)][f].loc[opt, "Mean"] - grand_mean_local)

    # Display
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.metric(label="Baseline S/N", value=f"{overall_sn:.4f} dB" if not np.isnan(overall_sn) else "N/A")
    with col2:
        st.metric(label="Predicted S/N (optimal)", value=f"{predicted_sn:.4f} dB" if not np.isnan(predicted_sn) else "N/A")
    with col3:
        st.metric(label=f"Grand mean ({resp})", value=f"{grand_mean_local:.4f}" if not np.isnan(grand_mean_local) else "N/A")

    opt_display = pd.DataFrame({
        "Factor": selected_factors,
        "Optimal (mapped)": [optimal_mapped[f] for f in selected_factors],
        "Optimal (coded)": [optimal_coded[f] for f in selected_factors]
    })
    st.table(opt_display)
    st.markdown(f"**Predicted mean response (optimal combo): {predicted_mean:.4f}**")

    # store for confirmatory table / export
    optimal_results[resp] = {"mapped": optimal_mapped, "coded": optimal_coded, "predicted_mean": predicted_mean,
                             "predicted_sn": predicted_sn, "grand_mean": grand_mean_local}

# Confirmatory value (uses last predicted_mean computed)
last_predicted_mean = list(optimal_results.values())[-1]["predicted_mean"] if optimal_results else np.nan
confirm_val = st.number_input("Enter confirmatory experimental value (optional)", value=float(last_predicted_mean) if not np.isnan(last_predicted_mean) else 0.0)
if not np.isnan(last_predicted_mean) and last_predicted_mean != 0:
    pred_err = abs(confirm_val - last_predicted_mean) / abs(last_predicted_mean) * 100.0
else:
    pred_err = np.nan
st.write(f"Confirmatory experimental value: {confirm_val:.4f}")
st.write(f"Prediction error: {pred_err:.2f} %" if not np.isnan(pred_err) else "Prediction error: N/A")

# -------------------------
# Confirmatory run table
# -------------------------
st.subheader("Confirmatory run (optimal levels)")
if optimal_results:
    # show the last one by default
    last_key = list(optimal_results.keys())[-1]
    last_opt = optimal_results[last_key]
    confirm_table = pd.DataFrame({
        "Factor": selected_factors,
        "Optimal (mapped)": [last_opt["mapped"][f] for f in selected_factors],
        "Optimal (coded)": [last_opt["coded"][f] for f in selected_factors]
    })
    st.dataframe(confirm_table)
else:
    st.info("No optimal results to show yet.")

# -------------------------
# Export full Excel report (raw original, mapped, per-factor sheets, SS, ANOVA tables, confirm)
# -------------------------
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    # Raw original
    try:
        original_df.to_excel(writer, sheet_name="Raw_original", index=False)
    except Exception:
        pd.DataFrame(original_df).to_excel(writer, sheet_name="Raw_original", index=False)
    # Raw mapped (editable)
    df_mapped.to_excel(writer, sheet_name="Raw_mapped", index=False)

    # per-response, per-factor tables
    for resp in selected_responses:
        for fac in selected_factors:
            tbl = results.get(resp, {}).get(fac)
            if tbl is not None and not tbl.empty:
                sheet = f"{resp}_{str(fac)}"[:31]
                try:
                    tbl.to_excel(writer, sheet_name=sheet)
                except Exception:
                    # fallback to safe name
                    tbl.to_excel(writer, sheet_name=f"{resp}_fac"[:31])

    # SS contribution
    try:
        ss_df.to_excel(writer, sheet_name="SS_Contribution")
    except Exception:
        pd.DataFrame(ss_df).to_excel(writer, sheet_name="SS_Contribution")

    # ANOVA tables
    for f, tbl in anova_tables.items():
        if tbl is not None:
            try:
                tbl.to_excel(writer, sheet_name=f"ANOVA_{str(f)}"[:31])
            except Exception:
                tbl.to_excel(writer, sheet_name=f"ANOVA_fac"[:31])

    # Optimal combos
    opt_rows = []
    for k, v in optimal_results.items():
        row = {"Response": k}
        for fac in selected_factors:
            row[f"Opt_{fac}"] = v["mapped"].get(fac, np.nan)
        row["Predicted_mean"] = v["predicted_mean"]
        row["Predicted_SN"] = v["predicted_sn"]
        opt_rows.append(row)
    if opt_rows:
        pd.DataFrame(opt_rows).to_excel(writer, sheet_name="Optimal_summary", index=False)

    # Summary
    summary_df = pd.DataFrame({
        "metric": ["Baseline S/N (dB)", "Predicted S/N (dB)", f"Grand mean ({selected_responses[0]})", "Predicted mean (optimal)", "Confirmatory value", "Prediction error %"],
        "value": [list(optimal_results.values())[0]["predicted_sn"] if optimal_results else np.nan,
                  list(optimal_results.values())[-1]["predicted_sn"] if optimal_results else np.nan,
                  grand_mean,
                  last_predicted_mean,
                  confirm_val,
                  pred_err]
    })
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

buf.seek(0)
st.download_button("📘 Download full Excel report", data=buf.getvalue(),
                   file_name="taguchi_report_v4.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("Done — all calculations updated. Edit the table, change mappings or weights, and the results will refresh automatically.")

# -------------------------
# ✅ Confirmatory Experiment Table
# -------------------------
st.header("✅ Confirmatory Experiment Suggestion")

confirm_df = pd.DataFrame([
    {"Response": resp,
     **{f"Optimal_{f}": opt["mapped"][f] for f in selected_factors},
     "Predicted Mean": opt["predicted_mean"],
     "Predicted S/N (dB)": opt["predicted_sn"]}
    for resp, opt in optimal_results.items()
])
st.dataframe(confirm_df.round(4))

# -------------------------
# 📤 Export all results to Excel
# -------------------------
st.header("📤 Export Results")

# Combine % Contribution + Optimal + ANOVA into one Excel
output_buffer = BytesIO()
with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
    ss_df.to_excel(writer, sheet_name="Percent_Contribution")
    for f, tbl in anova_tables.items():
        if tbl is not None:
            tbl.to_excel(writer, sheet_name=f"ANOVA_{str(f)[:20]}")
    for resp, opt in optimal_results.items():
        pd.DataFrame({
            "Factor": list(opt["mapped"].keys()),
            "Optimal (Mapped)": list(opt["mapped"].values()),
            "Optimal (Coded)": list(opt["coded"].values()),
            "Predicted Mean": [opt["predicted_mean"]] * len(opt["mapped"]),
            "Predicted S/N": [opt["predicted_sn"]] * len(opt["mapped"])
        }).to_excel(writer, sheet_name=f"Optimal_{resp[:20]}", index=False)
    df_mapped.to_excel(writer, sheet_name="Full_Data", index=False)

output_buffer.seek(0)
st.download_button(
    label="📥 Download all results as Excel",
    data=output_buffer,
    file_name="Taguchi_Advanced_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
