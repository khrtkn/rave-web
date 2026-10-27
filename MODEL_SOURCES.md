# 元モデルとブラウザー用変換

学習済みモデルは、作者が音の特徴を学習させたパラメーターを保存したものです。RAVEはその方式・実装の名前で、OrganやDjembeは個別の学習済みモデルです。本プロジェクトでは再学習せず、状態入出力・操作入力を追加します。

モデル本体は再配布しません。各自が公開元から取得してローカルで変換します。

## 作者・取得元・利用条件

| 対象 | 作者・提供元 | 元ファイル・取得先 | 条件の根拠 |
| --- | --- | --- | --- |
| Organ | Intelligent Instruments Lab | [b512 checkpoint / last.ckpt](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/resolve/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000/version_1/checkpoints/last.ckpt)・[config.gin](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/raw/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000/config.gin) | [モデルカード](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md)：CC BY-NC 4.0 |
| Djembe | Torsion Audio | [Scyclone / djembe.onnx](https://raw.githubusercontent.com/Torsion-Audio/Scyclone/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/assets/models/djembe.onnx) | [固定版Licenses節](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/README.md#licenses)：モデル用CC BY-NC 4.0 |

RAVEの研究・アーキテクチャーの著者は **Antoine Caillon / Philippe Esling、ACIDS–IRCAM**。[論文](https://arxiv.org/abs/2111.05011)・[公式ソース](https://github.com/acids-ircam/RAVE)。Organには **Victor Shepardson** の[RAVE fork](https://github.com/victor-shepardson/RAVE/tree/ab7d40d6740130f0268debbc4b6633318357ea7d)と[cached_conv fork](https://github.com/victor-shepardson/cached_conv/tree/d2da85992d030dcf1dd8ebf318c3d69753729295)の固定版を使用します。両forkもCC BY-NC 4.0です。

出典・ライセンス・変更点を保持し、非商用の範囲で利用してください。変換後も条件は変わりません。詳しい帰属表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にあります。

### 固定した入力

| ファイル | bytes | SHA-256 |
| --- | ---: | --- |
| Organ last.ckpt | 669,961,497 | `218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e` |
| Organ config.gin | 7,087 | `975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0` |
| Djembe djembe.onnx | 16,622,944 | `ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16` |

IIL revision：`c25a03d625840c40cd3a48779ed72f2a1947d7b4`。Scyclone commit：`2b1990c4ad04fec1981d21bdde94fdafbadb84e0`。取得後にサイズ・SHAを照合します。

## 取得 → 変換 → 検証 → 登録

Python 3.12でリポジトリーのルートから実行します。OrganにはGitと数GBの空き容量も必要です。NumPy等のversionが異なるため、モデルごとにvenvを分けます。

### Organ

```sh
python3.12 -m venv .venv-organ
.venv-organ/bin/python -m pip install -r outputs/rave-morph-lowlatency/model-tools/requirements-lock.txt
.venv-organ/bin/python scripts/prepare-models.py --model organ
```

公開checkpoint・設定・2つのforkを`work/organ-build/`へ取得します。weight normalizationを推論用にまとめ、36組の状態・ノイズ入力・PCA6方向のgain / biasを持つONNXを生成します。同じ重みで512／1024サンプル版を作ります。

元のnative処理・分割処理・ONNXの比較、座標操作、状態リセットを通過後、アプリに登録します。生成先は`outputs/rave-morph-lowlatency/web/models/organ-512/`と`organ-1024/`です。個別コマンドは [Organ model-tools](outputs/rave-morph-lowlatency/model-tools/README.md) を参照してください。

### Djembe

```sh
python3.12 -m venv .venv-djembe
.venv-djembe/bin/python -m pip install -r outputs/rave-morph/model-tools/requirements.txt
.venv-djembe/bin/python scripts/prepare-models.py --model djembe
```

固定ONNXを`work/djembe-build/djembe.onnx`へ取得します。39組の状態、ノイズ入力、4座標のgain / biasを追加。元の46 initializerを保持し、ゼロノイズ比較・操作・状態リセットを通過後に登録します。生成先は`outputs/rave-morph-lowlatency/web/models/djembe/`です。

追加シフトは1,360サンプル。通常の正規ノイズ比較には数値残差がある実験版です。Waterの中間モデルは不要です。個別コマンドは [Djembe model-tools](outputs/rave-morph/model-tools/DJEMBE.md) を参照してください。

### Windows

たとえばDjembeはコマンドプロンプトで次を実行します。

```bat
py -3.12 -m venv .venv-djembe
.venv-djembe\Scripts\python.exe -m pip install -r outputs/rave-morph/model-tools/requirements.txt
.venv-djembe\Scripts\python.exe scripts/prepare-models.py --model djembe
py -3.12 launch.py
```

Organは`.venv-organ`、Organのrequirements-lock、`--model organ`に置き換えます。Windows実機では未検証です。

### 起動と再検査

```sh
python3 launch.py
python3 scripts/model_catalog.py --verify
```

起動時にモデル・fixtureのSHAを照合します。Runtimeは公式npmの固定版から取得。登録一覧もGit管理対象外のローカル生成物です。片方だけ準備して起動でき、後から別のモデルを準備すると一覧に加わります。

`--work-dir PATH`と`--models-dir PATH`で別の場所に構築できます。`--models-dir`は個々のモデルディレクトリーの親です。別の場所へ出力した場合、付属アプリの既定保存先は変更されません。

## Waterは今回の手順に含めません

Water / Pond Brainの上流はIILの`water_pondbrain_b2048_r48000_z16.ts`（[モデルカード](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md)、CC BY-NC 4.0）です。以前の試作は別の中間stateful ONNXを直接改変していました。上流TorchScriptからその中間ONNXへの変換器・数値一致を再現していないため、元モデルを取得するだけで現在の変換器を通せるとは説明できません。過去のコードと検証資料は履歴資料として残しています。

## 確認範囲

実変換はmacOS / Python 3.12の既存依存環境で確認しました。Organ checkpointは固定SHAが一致するキャッシュを使用し、設定・forkは再取得しています。依存パッケージの新規一括インストール、Windows / Linux、すべてのブラウザーは未検証です。数値検証は聴感・継続再生・物理的な低遅延の保証ではありません。
