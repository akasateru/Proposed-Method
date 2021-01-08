ファイル名:             Proposed method 3.0　\n
説明:	                提案手法\n
情報源領域: 20news:     全文書\n
            dbpedia:    train.csv\n
            reuter:     全文書\n
            yahootopic: train_pu_half_v0.txt\n
                        train_pu_half_v1.txt\n
対象領域のクラス情報:   クラス名\n
学習データ選択方法:     rank1-rank2 > 0.05\n
                        各クラス文書数：3000\n
学習データ:             choiced_train_data.csv\n
テストデータ:           dbpedia/test.csv\n
パラメータ:             max_len = 128\n
                        units = 14\n
                        epocks = 30\n
                        batch_size = 4\n
