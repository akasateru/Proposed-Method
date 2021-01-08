ファイル名:             Proposed method 3.0
説明:	                提案手法
情報源領域: 20news:     全文書
            dbpedia:    train.csv
            reuter:     全文書
            yahootopic: train_pu_half_v0.txt
                        train_pu_half_v1.txt
対象領域のクラス情報:   クラス名
学習データ選択方法:     rank1-rank2 > 0.05
                        各クラス文書数：3000
学習データ:             choiced_train_data.csv
テストデータ:           dbpedia/test.csv
パラメータ:             max_len = 128
                        units = 14
                        epocks = 30
                        batch_size = 4
