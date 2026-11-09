import numpy as np

def torus_direction(src_x, src_y, target_x, target_y, field_size):
    diff_x = (target_x - src_x + field_size // 2) % field_size - field_size // 2
    diff_y = (target_y - src_y + field_size // 2) % field_size - field_size // 2
    return int(np.sign(diff_x)), int(np.sign(diff_y))

def run_tests():
    field_size = 10
    
    # テストケースの定義
    # (src_x, src_y, target_x, target_y, expected_dir_x, expected_dir_y, description)
    test_cases = [
        # 1. 同一地点 (移動不要)
        (3, 3, 3, 3, 0, 0, "同一地点"),
        
        # 2. 単純な近近移動 (ラップアラウンドなし)
        (3, 3, 5, 2, 1, -1, "右下方向（通常）"),
        (3, 3, 1, 5, -1, 1, "左上方向（通常）"),
        
        # 3. トーラス境界を跨ぐ移動 (ラップアラウンド)
        (1, 1, 9, 1, -1, 0, "左に巻く方が近い (1 -> 9)"),
        (9, 1, 1, 1, 1, 0, "右に巻く方が近い (9 -> 1)"),
        (1, 1, 1, 9, 0, -1, "上に巻く方が近い (1 -> 9)"),
        (1, 9, 1, 1, 0, 1, "下に巻く方が近い (9 -> 1)"),
        
        # 4. 境界値 (距離がちょうど field_size // 2)
        # field_size = 10 の場合、距離5は (-5 % 10 - 5) = -5 となり -1 方向として評価されます
        (0, 0, 5, 0, -1, 0, "ちょうど半周 (x方向)"),
        (0, 0, 0, 5, 0, -1, "ちょうど半周 (y方向)"),
    ]
    
    print("=== 個別テストケースの検証 ===")
    passed_count = 0
    for src_x, src_y, tgt_x, tgt_y, exp_x, exp_y, desc in test_cases:
        res_x, res_y = torus_direction(src_x, src_y, tgt_x, tgt_y, field_size)
        if (res_x, res_y) == (exp_x, exp_y):
            print(f"[PASS] {desc}")
            passed_count += 1
        else:
            print(f"[FAIL] {desc}: 期待値 ({exp_x}, {exp_y}), 取得値 ({res_x}, {res_y})")

    print(f"\n個別テスト結果: {passed_count}/{len(test_cases)} 通過")
    
    print("\n=== 全座標全探索テスト ===")
    # 愚直にマンハッタン距離を比較して最短方向を計算する判定器との突合
    errors = 0
    for sx in range(field_size):
        for sy in range(field_size):
            for tx in range(field_size):
                for ty in range(field_size):
                    # 期待される差分の計算（3x3の複製グリッドで最も近い距離を探す）
                    best_dx, best_dy = None, None
                    min_dist = float('inf')
                    
                    for ox in [-field_size, 0, field_size]:
                        for oy in [-field_size, 0, field_size]:
                            dx = (tx + ox) - sx
                            dy = (ty + oy) - sy
                            dist = abs(dx) + abs(dy)
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_dx, best_dy = dx, dy
                    
                    expected = (int(np.sign(best_dx)), int(np.sign(best_dy)))
                    actual = torus_direction(sx, sy, tx, ty, field_size)
                    
                    # 距離が調度半分（例: 距離5）の場合、タイブレークの挙動で符号が分かれることがありますが、
                    # 絶対値方向としては最短であることを確認
                    if actual != expected:
                        # タイブレーク判定（逆向きでも同じ距離ならOK）
                        alt_expected_x = -expected[0] if abs(tx - sx) == field_size / 2 else expected[0]
                        alt_expected_y = -expected[1] if abs(ty - sy) == field_size / 2 else expected[1]
                        
                        if actual != (alt_expected_x, alt_expected_y):
                            print(f"エラー検出: ({sx},{sy}) -> ({tx},{ty}) | 期待: {expected}, 取得: {actual}")
                            errors += 1

    if errors == 0:
        print("[SUCCESS] 全座標対での最短経路一致を確認しました。")
    else:
        print(f"[FAILURE] {errors} 件の不一致が検出されました。")

if __name__ == "__main__":
    run_tests()