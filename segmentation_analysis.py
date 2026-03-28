import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Customer_Churn_Prediction\Customer-Churn-Prediction-and-Segmentation\outputs\segmentation_results.csv")
num_col=data.select_dtypes(exclude="object").columns
summary=data.groupby("Segment")[num_col].mean()


plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x="tenure", y="MonthlyCharges", hue="Segment", palette="viridis")
plt.title("Customer Segments by Tenure and Charges")
plt.savefig("outputs/segment_scatterplot.png", bbox_inches="tight")
plt.show()

summary.to_csv("outputs/segmentation_summary_named.csv", index=True)

plt.savefig("outputs/churn_probability_by_segment.png", bbox_inches="tight")
print("scatter plot saved to outputs/churn_probability_by_segment.png")