# JBSI 解析アプリ

生体信号データ（CSV/TSV形式）を読み込み、フィルタリング・ベースライン補正・アーティファクト除去・イベント検出をインタラクティブに行うデスクトップアプリケーションです。

## 機能

- **ファイル読み込み** — CSV / TSV 形式の多チャンネル時系列データに対応
- **チャンネル選択** — 解析対象チャンネルをGUIで選択
- **Butterworth フィルタ** — ローパス / ハイパス / バンドパスに対応（SciPy 必須）
- **arPLS ベースライン補正** — 非対称最小二乗法による蛍光ベースライン除去（SciPy 必須）
- **アーティファクト除去** — スロープ・振幅閾値によるブロック検出 → 区間除去 → 時間軸圧縮。プレビューダイアログで視覚的にパラメータ調整可能
- **イベント検出** — Schmitt トリガー（z スコア閾値）による自動イベント検出、極性・不応期設定に対応
- **エクスポート**
  - PNG / PDF 保存
  - イベントリスト CSV 保存
  - GIF アニメーション出力（matplotlib 必須）

## スクリーンショット

> *(準備中)*

## 必要環境

| ライブラリ | 用途 | 必須 |
|---|---|---|
| Python 3.9+ | ランタイム | 必須 |
| PySide6 | GUI フレームワーク | 必須 |
| NumPy | 数値計算 | 必須 |
| pandas | ファイル読み込み | 必須 |
| SciPy | フィルタ / arPLS / アーティファクト除去 | 推奨 |
| pyqtgraph | 高速波形描画 | 推奨 |
| matplotlib | PNG / PDF / GIF エクスポート | 推奨 |

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/<your-username>/jbsi-analyzer.git
cd jbsi-analyzer

# 仮想環境を作成（任意）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージをインストール
pip install PySide6 numpy pandas scipy pyqtgraph matplotlib
```

## 使い方

```bash
python main.py
```

1. **「ファイルを開く」** ボタンで CSV / TSV ファイルを選択
   - 1列目: 時刻（秒）、2列目以降: 各チャンネルの信号値
2. チャンネル選択パネルで解析対象を選ぶ
3. フィルタ・ベースライン・アーティファクト除去のパラメータを設定
4. **「適用（処理実行）」** で解析を実行
5. 必要に応じて PNG / PDF / CSV / GIF でエクスポート

## 入力ファイル形式

```
time    ch1     ch2     ch3
0.000   0.12   -0.03    0.45
0.001   0.13   -0.02    0.44
...
```

- 区切り文字: タブ（`.tsv`）または カンマ（`.csv`）を自動判別
- ヘッダー行はチャンネル名として使用されます

## プロジェクト構成

```
app/
├── main.py               # エントリーポイント
└── app/
    ├── main_window.py    # メインウィンドウ（UI 全体）
    ├── models.py         # データモデル（RawModel, ProcessOptions, ProcessedResult）
    ├── processing.py     # 処理パイプライン（フィルタ・アーティファクト・正規化）
    ├── signal_proc.py    # 低レベル信号処理（Butterworth, z スコア）
    ├── baseline.py       # arPLS ベースライン推定
    ├── artifact.py       # アーティファクトブロック検出・拡張・マージ
    ├── events.py         # Schmitt トリガーイベント検出
    ├── worker.py         # バックグラウンド処理スレッド
    ├── io_utils.py       # ファイル読み込みユーティリティ
    ├── deps.py           # オプション依存ライブラリの管理
    └── views/
        ├── plot_area.py        # 波形描画エリア
        ├── column_selector.py  # チャンネル選択UI
        └── artifact_dialog.py  # アーティファクト調整ダイアログ
```

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。
