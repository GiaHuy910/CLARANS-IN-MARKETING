from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns

cmap=colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
def draw_correlation_matrix(data, features):
    corr = data[features].corr(numeric_only=True)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr,annot=True, cmap=cmap, center=0)
    
    plt.title('Correlation Matrix')
    plt.show()