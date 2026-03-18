import torch
from model.WPFormer import WPFormer

def run_shape_check():
    print("====== 1. 初始化模型 (WPFormer + ASER-Lite) ======")
    # 保持与你当前训练一致的配置
    model = WPFormer(method="pvt_v2_b2", channel=64)
    
    if torch.cuda.is_available():
        net = model.cuda()
        print("✅ CUDA 可用，模型已加载至 GPU。")
    else:
        net = model
        print("⚠️ 未检测到 CUDA，使用 CPU 运行。")
        
    net.eval()
    
    # 模拟 512x512 的 Batch 输入
    bc = 2
    dump_x = torch.randn(bc, 3, 512, 512)
    if torch.cuda.is_available():
        dump_x = dump_x.cuda()
    
    print("\n====== 2. 执行训练模式测试 (return_aux=True) ======")
    with torch.no_grad():
        preds, edge_logits, ambiguity_logits = net(dump_x, return_aux=True)
        
    print(f"主预测列表 (Preds) 长度: {len(preds)}")
    for i, p in enumerate(preds):
        if i == len(preds) - 1:
            print(f"  -> preds[{i}] (Refined Mask) shape: {p.shape}")
        elif i == len(preds) - 2:
            print(f"  -> preds[{i}] (Coarse Mask) shape: {p.shape}")
        else:
            print(f"  -> preds[{i}] shape: {p.shape}")
            
    print(f"\n辅助输出:")
    print(f"  -> Edge logits shape: {edge_logits.shape}")
    print(f"  -> Ambiguity logits shape: {ambiguity_logits.shape}")
    
    print("\n====== 3. 执行推理模式测试 (return_aux=False) ======")
    with torch.no_grad():
        preds_infer = net(dump_x, return_aux=False)
    print(f"推理模式下返回的 Preds 列表长度: {len(preds_infer)}")
    
    print("\n===================================================")
    print("🎯 Shape Check 完毕！如果没有报错且维度都是 [2, 1, 512, 512]，")
    print("说明模型结构和接口已完美闭环，可以去修改训练脚本的 Loss 并启动 Fold 1 了！")

if __name__ == '__main__':
    run_shape_check()