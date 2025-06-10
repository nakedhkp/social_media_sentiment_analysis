import csv
import random
from datetime import datetime, timedelta
import os

def generate_simulated_data(num_records=5000, output_file="data/raw/simulated_posts.csv"):
    """生成模拟的社交媒体数据"""
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    header = ['id', 'timestamp', 'user_id', 'text', 'platform']
    
    # 预设的情感文本模板
    positive_texts = [
        "这个产品真的很棒！强烈推荐给大家",
        "服务态度超级好，非常满意！",
        "质量很不错，物超所值",
        "客服很耐心，解决问题很及时",
        "包装精美，产品质量很好",
        "发货速度很快，很满意这次购物体验"
    ]
    
    negative_texts = [
        "质量太差了，完全不值这个价格",
        "客服态度很差，问题一直没解决",
        "发货太慢了，等了好久才收到",
        "产品和描述完全不符，很失望",
        "包装破损，产品也有问题",
        "售后服务太差了，不会再买了"
    ]
    
    neutral_texts = [
        "收到货了，还没来得及试用",
        "包装还可以，产品一般般",
        "价格适中，质量还行",
        "发货速度正常，产品符合预期",
        "客服回复及时，产品质量一般",
        "整体还可以，没有特别惊喜"
    ]
    
    platforms = ['微博', '知乎', '小红书', '抖音', '微信']
    
    all_data = []
    
    for i in range(num_records):
        post_id = f"post_{i:06d}"
        
        # 生成过去30天内的随机时间
        timestamp = (datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )).strftime("%Y-%m-%d %H:%M:%S")
        
        user_id = f"user_{random.randint(1, 1000)}"
        platform = random.choice(platforms)
        
        # 按比例生成不同情感的文本
        rand_choice = random.random()
        if rand_choice < 0.4:  # 40% 积极
            text = random.choice(positive_texts)
        elif rand_choice < 0.7:  # 30% 消极
            text = random.choice(negative_texts)
        else:  # 30% 中性
            text = random.choice(neutral_texts)
        
        all_data.append([post_id, timestamp, user_id, text, platform])
    
    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)
    
    print(f"✅ 成功生成 {num_records} 条模拟数据到 {output_file}")
    return output_file

if __name__ == "__main__":
    generate_simulated_data(5000)