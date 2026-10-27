# Djembe の元モデル取得とローカル変換

この手順は、公開元の学習済み ONNX を自分のPCへ取得し、ブラウザーで過去状態を引き継げる ONNX に変換します。再学習はしません。Water の中間モデルや、作者のローカル開発フォルダーは不要です。

## 元モデル・作者・条件

- 提供元：Torsion Audio。RAVE の研究・アーキテクチャー：Antoine Caillon / Philippe Esling、ACIDS–IRCAM。
- 元ファイル：[Scyclone の djembe.onnx](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/assets/models/djembe.onnx)。固定コミット：`2b1990c4ad04fec1981d21bdde94fdafbadb84e0`。
- サイズ：16,622,944 bytes。SHA-256：`ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16`。
- モデルの条件：[固定版 README の Licenses 節](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/README.md#licenses) に記載された [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。出典・ライセンス・変更の表示と非商用条件を守ってください。ONNX 変換によって条件が解除されるものではありません。

## 実行

リポジトリーのルートで実行します。Python 3.12 と、同じディレクトリーの `requirements.txt` を使用します。

```sh
python3.12 -m venv .venv-models
.venv-models/bin/python -m pip install -r outputs/rave-morph/model-tools/requirements.txt
.venv-models/bin/python outputs/rave-morph/model-tools/prepare_djembe_source.py --output work/djembe-source/djembe.onnx
.venv-models/bin/python outputs/rave-morph/model-tools/build_djembe_candidate.py --source work/djembe-source/djembe.onnx --out work/generated-models/djembe
.venv-models/bin/python outputs/rave-morph/model-tools/audit_djembe_candidate.py --source work/djembe-source/djembe.onnx --out work/generated-models/djembe
```

Windows では `.venv-models/bin/python` を `.venv-models\Scripts\python.exe` に読み替えます。確認済み環境は macOS / Python 3.12 です。

取得 helper は固定 URL のみを使い、サイズ・SHA-256 の照合が済むまで最終ファイルを作りません。途中の失敗では一時ファイルを取り除き、既存の不一致ファイルや symlink を上書きしません。URL のリダイレクトも拒否します。

生成先は指定した `--out` です。モデル、契約、Djembe 単独 catalog、39 組の状態の対応、合成音声による parity fixture、検証レポートを出力します。共通 catalog・Water の成果物・元モデルは書き換えません。モデルバイナリーを GitHub に再配布する手順ではありません。

## 検証と限界

- 固定元 ONNX と生成 ONNX の native 比較：ゼロノイズ・同一合成入力を使用し、既知の追加シフト 1,360 samples と初期区間を除いて照合します。
- 元の全46 initializer の byte 一致、全音声・状態の有限値、reset 後の出力・全状態の再現、4軸それぞれの gain / bias の作用を確認します。
- 保存済み fixture を別プロセスで再読込し、生成モデルの native 出力と照合します。
- 標準正規ノイズでの比較には float32 の残差があるため、ゼロノイズの厳密な基準を通ったことを全条件での完全一致とは扱いません。詳細は [モデル別の説明](../web/models/djembe-candidate/README.md) を参照してください。
- 1,360 samples は改変で加わった相対的な遅れです。元モデル自体の絶対的な遅れ、ブラウザーの速度、マイクからスピーカーまでの遅延や連続再生は保証しません。

取得 helper の異常系テスト：

```sh
python3 -m unittest discover -s outputs/rave-morph/model-tools -p test_djembe_source.py -v
```
