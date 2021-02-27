- ファイル名:
    - Proposed method 16.0
- 説明:
    - 情報源領域にdbpedia/test.csvのみ使用した場合
- 情報源領域:
    <!-- - 20news:全文書 -->
    - dbpedia:train.csv
    <!-- - reuter:全文書 -->
    <!-- - yahootopic:train_pu_half_v0.txt, train_pu_half_v1.txt -->
- 対象領域のクラス情報:
    - クラス名
- 学習データ選択方法:
    - rank1-rank2 > 0.05
- 各クラス文書数：3000
- 5と9の文書数：3000
- 学習データ:
    - choiced_train_data.csv
    <!-- - yahootopic/train_pu_half_v1.txt -->
- テストデータ:
    - dbpedia/test.csv
    <!-- - yahootopic/test.txt -->
- パラメータ:
    - max_len = 128
    - units = 14
    - epochs = 20
    - batch_size = 4
