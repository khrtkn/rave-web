# RAVE Web

公開された学習済みRAVEモデルを、自分のPCでブラウザー用ONNXへ変換して使う研究プロトタイプです。音声処理はブラウザー内で実行します。

**配布内容：アプリのソース、元モデルの取得・変換・検証スクリプト、出典と利用条件。モデル本体はGitにもReleaseにも同梱しません。Webサービスの配信も行いません。**

## 何を作ったのか

RAVEは、音を小さな数値の表現に圧縮し、そこから音を作り直すニューラル音声モデルです。OrganやDjembeは、それぞれ別の音の特徴を学習済みのモデルです。この実装では再学習せず、過去の音の状態を引き継ぐ入出力と音色操作を加え、ONNX Runtime Webで動かします。

```text
公開元の学習済みモデル → 各自のPCで取得 → ONNXへ変換・数値検証
                                              ↓
デモ／音声ファイル／マイク → ブラウザー内のRAVE → 音声出力
```

Neutone Morphoとは独立した実装です。Morphoのエンジン・モデル・バイナリーは含みません。各モデル作者による公式ブラウザー版でもありません。

## 最初の準備

必要環境は **Python 3.12**、Chrome / Edgeなどの対応ブラウザーです。Organの変換にはGitも必要です。[ZIP](https://github.com/khrtkn/rave-web/archive/refs/heads/main.zip) またはGitでコードを取得してください。

```sh
git clone https://github.com/khrtkn/rave-web.git
cd rave-web
```

作者・固定取得先・利用条件・詳しい手順は **[MODEL_SOURCES.md](MODEL_SOURCES.md)** にあります。まず次のどちらかを準備します。

### Organ：主な研究対象

元checkpointは約670MB。Python依存関係を含め数GBの空き容量が必要です。リポジトリーのルートで実行します。

```sh
python3.12 -m venv .venv-organ
.venv-organ/bin/python -m pip install -r outputs/rave-morph-lowlatency/model-tools/requirements-lock.txt
.venv-organ/bin/python scripts/prepare-models.py --model organ
python3 launch.py
```

同じ重みから512／1024サンプル版を生成し、数値検証を通過したモデルを登録します。

### Djembe：小型の実験モデル

元ONNXは約16.6MB。Organとは別の依存環境を使います。

```sh
python3.12 -m venv .venv-djembe
.venv-djembe/bin/python -m pip install -r outputs/rave-morph/model-tools/requirements.txt
.venv-djembe/bin/python scripts/prepare-models.py --model djembe
python3 launch.py
```

**Windows**：環境作成は`py -3.12 -m venv .venv-djembe`、Pythonのパスは`.venv-djembe\Scripts\python.exe`のように置き換えます。[Windows手順](MODEL_SOURCES.md#windows)を参照してください。

最初の起動は公式npmからONNX Runtime Web 1.30.0を約33MB取得します。モデル準備後の起動にはPython 3.9以上を使えます。`Start-macOS.command`／`Start-Windows.bat`も、モデル準備後に使用します。

ブラウザーが開いたらモデルをロードし、低い音量から内蔵デモを試してください。起動したターミナルを開いたまま使用し、終了時は`Ctrl+C`。GitHubのページだけでは音は鳴りません。

## モデルと実装

| 元モデル | 提供元 | ローカル変換 | 操作 |
| --- | --- | --- | --- |
| Organ b512 checkpoint | Intelligent Instruments Lab | 512／1024版を生成・native検証 | 保存済みPCA先頭6方向 |
| Djembe ONNX | Torsion Audio / Scyclone | 2048版を生成・native検証。実験版 | 保持された4座標 |
| Water / Pond Brain | Intelligent Instruments Lab | **今回の再現手順には含めない** | 過去の比較実験のみ |

Waterは以前の中間ONNXに依存し、公開元TorchScriptからの変換経路と数値一致を再現できていません。履歴資料・スクリプトは残しますが、今回準備できるモデルとしては表示しません。

Web Audio / AudioWorklet → Worker → ONNX Runtime Web → 状態を引き継ぐRAVE encoder / decoderという処理経路です。WASMと、対応環境ではWebGPUを選べます。潜在座標のgain / bias、Dry / Wet、音量を操作できます。

音声ファイル・マイク入力はブラウザー内で扱います。Pythonサーバーは`127.0.0.1`上の静的配信のみです。音声処理API・アップロードAPIはありません。SharedArrayBuffer用の分離ヘッダーを設定するため、`file://`でHTMLを直接開かず、付属ランチャーを使ってください。

## 再現性と確認範囲

- 取得元・固定revision・サイズ・SHA-256を固定し、取得後に照合します。変換後は元モデルとの比較、状態リセット、座標操作、合成入力fixtureを検証します。
- Organ 512／1024とDjembeは別フォルダーで実変換を確認しました。Organの約670MB checkpointは既存キャッシュを固定SHAで再検証し、設定とforkは再取得しました。Djembeとnpm runtimeは実ダウンロードを確認しました。
- Python依存関係は既存の検証環境を使用しました。新しいPCへの依存パッケージ一括インストールは未検証です。macOSでの記録で、Windows / Linuxの実機確認はしていません。
- 数値一致は、聴感上の品質、低遅延、マイクの連続再生を保証しません。実マイクでは音切れが残ります。
- Djembeには追加シフト1,360サンプルがあり、正規ノイズの比較には数値残差があります。元モデル自体の絶対的な遅延は未測定です。
- 512サンプル／48kHzの10.67msは処理する音の長さで、マイクからスピーカーまでの遅延ではありません。

## 構成と検証

```text
MODEL_SOURCES.md                  作者・取得元・条件・変換手順
scripts/prepare-models.py          取得 → 変換 → 検証 → ローカル登録
scripts/model_catalog.py           生成モデル・fixtureの整合性検査
scripts/fetch-assets.py            公式npmからRuntimeのみ取得
launch.py / serve.py               ローカル起動
outputs/rave-morph-lowlatency/     現在のアプリとOrgan変換器
outputs/rave-morph/                Djembe変換器と過去の比較実験
```

元のディレクトリー名を維持しています。生成モデル、元checkpoint、ローカル検証結果、録音、venvはGitの対象外です。過去の比較実装のfixtureは人工的なテスト入力で、個人の録音ではありません。

```sh
node --test outputs/rave-morph-lowlatency/tests/*.test.mjs outputs/rave-morph/tests/*.test.mjs
python3 -m unittest discover -s scripts/tests -v
python3 -m unittest discover -s outputs/rave-morph/model-tools -p test_djembe_source.py -v
python3 scripts/model_catalog.py --verify
python3 scripts/fetch-assets.py --verify
```

最後の2つはモデルとRuntimeを準備した後に実行します。

## ライセンス

独自の制作物は [CC BY-NC 4.0](LICENSE) です。元モデルも非商用条件付きです。**各自で取得する方式にしても、非商用・出典表示・変更表示などの条件は引き続き適用されます。** 商用利用の追加許諾を付与するものではありません。

ONNX Runtime WebはMIT Licenseの別作品です。作者・取得元・適用範囲は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
