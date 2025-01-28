# AutoLP - セミナータイトルジェネレーター

セミナーのタイトル、見出し、本文をAIを使って自動生成するツールです。

## 機能

1. セミナータイトル生成
- メインタイトルとサブタイトルの2部構成
- タイトルの評価・改善提案
- 集客速度予測

2. 見出し生成
- 3つの見出し（背景・課題・解決策）
- 手動編集機能

3. 本文生成
- 見出しに基づく本文生成
- マークダウンフォーマット

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/hamazinger/AutoLP.git
cd AutoLP

# 依存関係のインストール
pip install -r requirements.txt
```

## 環境設定

`.env`ファイルを作成し、以下の変数を設定してください：

```env
OPENAI_API_KEY=your_openai_api_key
GCP_SERVICE_ACCOUNT={
  "type": "service_account",
  ...
}
```

## 使い方

```bash
# アプリケーションの起動
streamlit run src/main.py
```

## プロジェクト構造

```
src/
├── __init__.py
├── main.py                    # アプリケーションのエントリーポイント
├── config/                   # 設定管理
├── models/                   # データモデル
├── services/                 # ビジネスロジック
├── data/                     # データアクセス層
├── utils/                    # ユーティリティ
└── ui/                       # ユーザーインターフェース
```

## 開発フロー

1. `.env`ファイルの設定
2. 依存関係のインストール
3. `streamlit run src/main.py`でアプリケーションを起動
4. http://localhost:8501 でアクセス

## カスタマイズ

1. プロンプトの編集
- `utils/prompts.py`でプロンプトテンプレートを編集
- UI上でもプロンプトの動的な編集が可能

2. 評価基準のカスタマイズ
- `services/evaluator.py`で評価基準やスコアリングをカスタマイズ

3. UIのカスタマイズ
- `ui/components/`以下のコンポーネントを編集

## データモデル

1. タイトル関連
- `GeneratedTitle`: 生成されたセミナータイトル
- `TitleEvaluation`: タイトルの評価結果
- `TitleAnalysis`: 詳細なタイトル分析結果

2. コンテンツ関連
- `WebContent`: Webページから抽出したコンテンツ
- `HeadlineSet`: 生成された見出しセット

## 依存関係

```txt
streamlit>=1.24.0
langchain>=0.1.7
langchain-community>=0.3.14
langchain-openai>=0.3.0
google-cloud-bigquery>=3.11.0
pandas-gbq>=0.19.2
db-dtypes>=1.1.1
openai>=1.6.1
pandas>=2.0.0
numpy>=1.24.0
beautifulsoup4==4.12.2
trafilatura>=1.5.0
PyPDF2>=3.0.0
python-docx>=0.8.11
requests>=2.28.0
pydantic>=2.6.3
```

## ライセンス

MIT License

## 貢献

1. Forkする
2. ブランチを作成する (`git checkout -b feature/amazing_feature`)
3. 変更をコミットする (`git commit -m 'Add some amazing feature'`)
4. ブランチをプッシュする (`git push origin feature/amazing_feature`)
5. Pull Requestを作成する
