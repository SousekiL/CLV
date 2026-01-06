# 加载必要的库
library(tidyverse)
library(lubridate)
library(CLVTools)

# =======================================================
# 步骤 0: 环境准备与数据加载
# =======================================================
# 请根据实际情况修改路径
file_path <- "/Users/sousekilyu/Downloads/online_retail_II.xlsx"

df_0910 <- readxl::read_xlsx(file_path, sheet = "Year 2009-2010") %>%
  as_tibble()
df_1011 <- readxl::read_xlsx(file_path, sheet = "Year 2010-2011") %>%
  as_tibble()
df_raw <- bind_rows(df_0910, df_1011)

# =======================================================
# 步骤 1: 严格的数据清洗 (Data Cleaning)
# =======================================================
df_cleaned <- df_raw %>%
  filter(!is.na(`Customer ID`)) %>% # 必须有ID
  filter(!str_detect(Invoice, "^C")) %>% # 剔除退货订单
  filter(Quantity > 0 & Price > 0) %>% # 剔除异常数值
  mutate(
    TotalRevenue = Quantity * Price,
    InvoiceDate = as.Date(InvoiceDate) # 统一日期格式
  ) %>%
  select(`Customer ID`, InvoiceDate, TotalRevenue)

# 获取全量用户名单用于最后合并
all_unique_customers <- unique(df_cleaned$`Customer ID`)

# =======================================================
# 步骤 2: 构建全量 CLV 数据对象
# =======================================================
clv_data_full <- clvdata(
  data.transactions = df_cleaned,
  date.format = "%Y-%m-%d",
  time.unit = "weeks",
  estimation.split = NULL, # 使用全部数据进行训练
  name.id = "Customer ID",
  name.date = "InvoiceDate",
  name.price = "TotalRevenue"
)

print("数据对象构建完成。")

# =======================================================
# 步骤 3: 拟合模型
# =======================================================

# 3.1 拟合 BG/NBD 模型
print("正在拟合 BG/NBD 模型...")
est_bgnbd_full <- bgnbd(
  clv_data_full,
  verbose = FALSE,
  optimx.args = list(method = "Nelder-Mead")
)

# 3.2 拟合 Gamma-Gamma 模型
print("正在拟合 Gamma-Gamma 模型...")
est_gg_full <- gg(clv_data_full, verbose = FALSE)

# =======================================================
# 步骤 4: 全量预测
# =======================================================
print("正在进行全量预测...")

# 预测未来 52 周 (1年)
final_clv_table <- predict(
  est_bgnbd_full,
  prediction.end = 52,
  continuous.discount.factor = 0.1,
  predict.spending = est_gg_full
)

# =======================================================
# 步骤 5: 构建最终交付表
# =======================================================

# 转为 Dataframe
result_df <- as.data.frame(final_clv_table)

# 调试：再次打印列名确认
print("预测结果列名：")
print(colnames(result_df))

# FIX: 提取历史统计信息 (Frequency)
# 由于部分版本 CLVTools 对象结构不同，直接访问 @cbs 可能报错
# 我们直接从清洗后的源数据计算：Frequency = (独立交易日期数) - 1
historical_stats <- df_cleaned %>%
  group_by(`Customer ID`) %>%
  summarise(frequency = n_distinct(InvoiceDate) - 1) %>%
  ungroup() %>%
  rename(Id = `Customer ID`) %>%
  mutate(Id = as.character(Id))

# 准备全量用户表
all_customers_df <- data.frame(Id = as.character(all_unique_customers))

# 合并结果
final_output <- all_customers_df %>%
  left_join(result_df, by = "Id") %>%
  left_join(historical_stats, by = "Id") %>% # FIX: 这里的 left_join 把 frequency 加进来了
  rename(CustomerID = Id) %>%
  mutate(
    # FIX: 根据实际列名进行映射
    # CET: Conditional Expected Transactions (预测交易次数)
    # predicted.period.spending: 预测期间的总价值 (CLV)

    CLV_1Year = replace_na(predicted.period.spending, 0),
    Predicted_Transactions = replace_na(CET, 0),
    Predicted_Mean_Spending = replace_na(predicted.mean.spending, 0),
    P_Alive = replace_na(PAlive, 0),
    frequency = replace_na(frequency, 0) # 填充可能的 NA
  ) %>%
  arrange(desc(CLV_1Year)) %>%
  select(
    CustomerID,
    period.first, # 第一次购买时间
    period.last, # 最后一次购买时间
    frequency, # 历史复购次数
    P_Alive, # 存活概率
    Predicted_Transactions, # 预测未来交易次数
    Predicted_Mean_Spending, # 预测客单价
    CLV_1Year # 预测 CLV (总价值)
  )

# =======================================================
# 步骤 6: 结果展示与导出
# =======================================================

print("前 10 名高价值客户预测：")
print(head(final_output, 10))

# 导出 CSV
write_csv(final_output, "Customer_Lifetime_Value_Prediction_Final.csv")

print("完成！数据已导出。")


# =======================================================
# 步骤 7: 可视化 - 高价值客户：历史 vs 预测对比
# =======================================================
print("正在生成高价值客户对比图...")

# 选取 Top 15 高价值客户
top_customers_plot_data <- final_output %>%
  head(15) %>%
  select(CustomerID, frequency, Predicted_Transactions) %>%
  pivot_longer(
    cols = c(frequency, Predicted_Transactions),
    names_to = "Type",
    values_to = "Transactions"
  ) %>%
  mutate(
    Type = ifelse(
      Type == "frequency",
      "Historical (Past)",
      "Predicted (Future 1Y)"
    )
  )

# 绘制条形图
p1 <- ggplot(
  top_customers_plot_data,
  aes(x = reorder(CustomerID, Transactions), y = Transactions, fill = Type)
) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Top 15 CLV Customers: Historical vs Predicted Transactions",
    subtitle = "Comparing past purchase frequency with expected future behavior",
    x = "Customer ID",
    y = "Number of Transactions",
    fill = "Time Period"
  ) +
  theme_minimal() +
  scale_fill_manual(
    values = c(
      "Historical (Past)" = "#7f8c8d",
      "Predicted (Future 1Y)" = "#e74c3c"
    )
  )

# 保存图片
ggsave("Top15_Customers_Comparison.png", plot = p1, width = 10, height = 8)
print("已保存: Top15_Customers_Comparison.png")


# =======================================================
# 步骤 8: 可视化 - 回测验证 (与“未来”实际数据对比)
# =======================================================
# 逻辑：为了模拟“预测 vs 实际”，我们把时间轴回拨。
# 必须先筛选出在“训练期”就已经存在的客户，否则 CLVTools 会报错，
# 因为它不能为训练期没有数据的“未来客户”建模。

print("正在进行回测验证 (Back-testing)...")

# 8.1 确定分割时间点 (保留最后 12 周作为验证期)
last_date <- max(df_cleaned$InvoiceDate)
split_date <- last_date - weeks(12)

print(paste0("回测分割点: ", split_date))

# 8.2 [关键步骤] 筛选有效的回测用户
# 找出所有在分割点之前就有“第一次购买”行为的客户 ID
valid_cohort_customers <- df_cleaned %>%
  group_by(`Customer ID`) %>%
  summarise(first_purchase = min(InvoiceDate)) %>%
  filter(first_purchase <= split_date) %>%
  pull(`Customer ID`)

# 仅保留这些有效客户的交易记录用于回测
df_holdout_subset <- df_cleaned %>%
  filter(`Customer ID` %in% valid_cohort_customers)

print(paste0("回测有效客户数: ", length(valid_cohort_customers)))

# 8.3 创建带 Holdout 的数据对象 (使用筛选后的子集)
clv_data_holdout <- clvdata(
  data.transactions = df_holdout_subset, # 注意这里使用 subset
  date.format = "%Y-%m-%d",
  time.unit = "weeks",
  estimation.split = split_date,
  name.id = "Customer ID",
  name.date = "InvoiceDate",
  name.price = "TotalRevenue"
)

# 8.4 在训练集上拟合临时模型
print("正在拟合回测模型...")
est_bgnbd_holdout <- bgnbd(
  clv_data_holdout,
  verbose = FALSE,
  optimx.args = list(method = "Nelder-Mead")
)

# 8.5 绘制 "预测 vs 实际" 累积交易曲线 (Tracking Plot)
# 这一步 CLVTools 会自动计算 Holdout 区间的真实购买数，并与模型预测数做对比
# 红色虚线(Model) 越贴近 黑色实线(Actual)，说明模型越准
png("Model_Validation_Actual_vs_Predicted.png", width = 800, height = 600)
plot(
  est_bgnbd_holdout,
  cumulative = TRUE,
  plot = TRUE,
  label = "Model Fit Visualization"
)
title(main = "Validation: Actual vs Predicted Transactions (Holdout Period)")
dev.off()

print("已保存: Model_Validation_Actual_vs_Predicted.png")
print("分析全部完成！")
