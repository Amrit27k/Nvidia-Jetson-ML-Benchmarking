import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def plot_power_vs_fps(data, plot_title, file_name, device):
    """
    Plots Power vs FPS for each model in the given data.

    Args:
        data: A pandas DataFrame containing the data with columns 'Model', 'Total Time', 'Achieved F', 'Avg Inferer', 'Avg Power', 'Device'.
    """
    device_data = data[data['Device'] == device]
    # Create a new figure with subplots for each model
    plt.figure(figsize=(10, 6))
    # Get the list of unique models
    models = device_data['Model'].unique()
    # Iterate over each model
    
    for model in models:
        model_data = device_data[device_data['Model'] == model]
        plt.plot(model_data['Achieved FPS'], model_data['Avg Power Consumption'], marker='o', label=model)

    plt.xlabel('FPS (Frames per Second)')
    plt.ylabel('Power Consumption (mW)')
    plt.title(plot_title)
    plt.legend(title='Model')
    plt.grid(True)
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(file_name, format='png', dpi=300)
    return device_data  

def dev_polynomial_eqn(filter_data: dict):
    models = filter_data['Model'].unique()
    plt.figure(figsize=(10, 6))
    for model in models:
        model_data = filter_data[filter_data['Model'] == model]
        x = model_data['Achieved FPS']
        y = model_data['Avg Power Consumption']
        
        #specify degree of 3 for polynomial regression model
        #include bias=False means don't force y-intercept to equal zero
        poly = PolynomialFeatures(degree=3, include_bias=False)
            
        #reshape data to work properly with sklearn
        poly_features = poly.fit_transform(x.values.reshape(-1, 1))
            
        #fit polynomial regression model
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
            
        #display model coefficients
        print(poly_reg_model.intercept_, poly_reg_model.coef_)
        y_predicted = poly_reg_model.predict(poly_features)
            
        #create scatterplot of x vs. y
        # plt.scatter(x, y)
            
        #add line to show fitted polynomial regression model
        plt.plot(x, y_predicted, label=model)

    plt.xlabel('FPS (Frames per Second)')
    plt.ylabel('Power Consumption (mW)')
    plt.legend(title='Model')
    plt.grid(True)
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Assuming your data is in a CSV file named 'data.csv'
data = pd.read_csv('/kaggle/input/fps-power-orin/output_data_fps.csv')
# Plot the power vs fps graph
filtered_data_device = plot_power_vs_fps(data, "Power Consumption vs FPS (min to 1) for Orin", "test_gpu", "GPU")
dev_polynomial_eqn(filtered_data_device)


