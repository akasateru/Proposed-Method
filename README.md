- ファイル名:
    - Proposed method 5.1
- 説明:
    - 情報源にほかのデータセットを入れない
    - 閾値を0.01に変更
- 情報源領域:
    - dbpedia:train.csv
- 対象領域のクラス情報:
    - クラス名
- 学習データ選択方法:
    - rank1-rank2 > 0.01
    - 各クラス文書数：3000
- 学習データ:
    - choiced_train_data.csv
- テストデータ:
    - dbpedia/test.csv
- パラメータ:
    - max_len = 128
    - units = 14
    - epochs = 20
    - batch_size = 4
