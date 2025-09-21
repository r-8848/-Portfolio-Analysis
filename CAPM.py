import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

BSE_PATH = "bsedata1.csv"
NSE_PATH = "nsedata1.csv"

def market_risk_and_return(path):
    df = pd.read_csv(path)
    daily_returns = np.array(df[df.columns[-1]])

    df = pd.DataFrame(np.transpose(daily_returns))
    m, sigma = np.mean(df, axis = 0) * len(df) / 5, df.std()

    mu_market = m[0]
    risk_market = sigma[0]

    print("Annualized Market return =", mu_market*100,"%")
    print("Market risk =", risk_market*100, "%")

    return mu_market, risk_market


def plot_single_curve(x, y, x_axis_label, y_axis_label, title):
    plt.plot(x, y, color='navy')
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label) 
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_dual_curves(x1, y1, x2, y2, x_axis_label, y_axis_label, title, x_market, y_market):
    plt.plot(x1, y1, color='blue', label='Minimum Variance Curve')
    plt.plot(x2, y2, color='green', label='CML')
    plt.scatter(x_market, y_market, color='orange', label='Market Risk and Return', marker='o', s=100)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label) 
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def minimum_variance_portfolio(m, C):
    C_inv = np.linalg.inv(C)
    u = np.ones(len(m))
    uT = np.transpose(u)
    mT = np.transpose(m)

    weight_min_var = u @ C_inv / (u @ C_inv @ uT)
    mu_min_var = weight_min_var @ mT
    risk_min_var = math.sqrt(np.dot(np.dot(weight_min_var,C),np.transpose(weight_min_var)))

    return risk_min_var, mu_min_var


def minimum_variance_line(m, C, mu):
    C_inv = np.linalg.inv(C)
    u = np.ones(len(m))
    uT = np.transpose(u)
    mT = np.transpose(m)

    p = [[1, u @ C_inv @ mT], [mu, m @ C_inv @ mT]]
    q = [[u @ C_inv @ uT, 1], [m @ C_inv @ uT, mu]]
    r = [[u @ C_inv @ uT, u @ C_inv @ mT], [m @ C_inv @ uT, m @ C_inv @ mT]]

    det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)

    w = (det_p * (u @ C_inv) + det_q * (m @ C_inv)) / det_r
    var = math.sqrt(np.dot(np.dot(w, C), np.transpose(w)))

    return w, var

def plot_efficient_frontier(m, C, risk_free_rate):
    returns = np.linspace(-3, 5, num = 2000)
    u = np.ones(len(m))
    risk = []

    for mu in returns:
        _,var = minimum_variance_line(m, C, mu)
        risk.append(var)

    risk_min_var, mu_min_var = minimum_variance_portfolio(m,C)

    higher_returns, higher_risk, lower_returns, lower_return_risk = [], [], [], []
    for i in range(len(returns)):
        if returns[i] >= mu_min_var: 
            higher_returns.append(returns[i])
            higher_risk.append(risk[i])
        else:
            lower_returns.append(returns[i])
            lower_return_risk.append(risk[i])

    market_weights = (m - risk_free_rate * u) @ np.linalg.inv(C) / ((m - risk_free_rate * u) @ np.linalg.inv(C) @ np.transpose(u) )
    mu_market = market_weights @ np.transpose(m)
    risk_market = math.sqrt(market_weights @ C @ np.transpose(market_weights))

    plt.plot(higher_risk, higher_returns, color='mediumseagreen', label='Efficient frontier')  
    plt.plot(lower_return_risk, lower_returns, color='royalblue') 
    plt.xlabel("Risk")
    plt.ylabel("Returns")
    plt.title("Minimum Variance Curve & Efficient Frontier")
    
    plt.plot(risk_market, mu_market, color='orange', marker='o')  
    plt.annotate(f'Market Portfolio\n(Risk: {round(risk_market, 2)}, Return: {round(mu_market, 2)})',
             xy=(risk_market, mu_market), xytext=(0.2, 0.6),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.plot(risk_min_var, mu_min_var, color='orange', marker='o') 
    plt.annotate(f'Min. Variance Portfolio\n(Risk: {round(risk_min_var, 2)}, Return: {round(mu_min_var, 2)})',
             xy=(risk_min_var, mu_min_var), xytext=(risk_min_var, -0.6),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.legend()
    plt.grid(True)
    plt.show()

    print()
    print("Market Portfolio Weights = ", market_weights)
    print("Return = ", mu_market)
    print("Risk = ", risk_market * 100, " %")

    return mu_market, risk_market


def plot_capital_line(m, C, risk_free_rate, mu_market, risk_market):
    returns = np.linspace(-3, 5, num = 2000)
    u = np.ones(len(m))
    risk = []

    for mu in returns:
        w,sigma = minimum_variance_line(m, C, mu)
        risk.append(sigma)

    returns_cml = []
    risk_cml = np.linspace(0, 1, num = 2000)
    for i in risk_cml:
        returns_cml.append(risk_free_rate + (mu_market - risk_free_rate) * i / risk_market)

    slope, intercept = (mu_market - risk_free_rate) / risk_market, risk_free_rate
    print()
    print("Equation of Capital Market Line is:")
    print(f"y = {slope:.2f} x + {intercept:.2f}\n")

    plot_dual_curves(risk, returns, risk_cml, returns_cml, "Risk", "Returns", "Capital Market Line with Markowitz Efficient Frontier", risk_market,mu_market)
    plot_single_curve(risk_cml, returns_cml, "Risk", "Returns", "Capital Market Line")


def plot_security_line(risk_free_rate, mu_market):
    beta_k = np.linspace(-1, 1, 5000)
    mu_k = risk_free_rate + (mu_market - risk_free_rate) * beta_k
    print("Equation of Security Market Line is:",f"mu = {mu_market - risk_free_rate:.2f} beta + {risk_free_rate:.2f}")
    print()
    plt.plot(beta_k, mu_k)
    plt.title('Security Market Line for the 10 assets')
    plt.xlabel("Beta")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.show()

  
def q2a(path, mu_market_index, risk_market_index):
    df = pd.read_csv(path)
    df.set_index('Date',inplace=True)
    df_index = df[df.columns[0:10]]
    df_nonindex = df[df.columns[10:20]]

    m = np.mean(df_index,axis=0)*len(df)/5
    # print(m)
    C = df_index.cov()
    # print(C)
    mu_market,risk_market = plot_efficient_frontier(m,C,0.05)
    plot_capital_line(m,C,0.05,mu_market,risk_market)
    plot_security_line(mu_market_index,risk_market_index)


    m = np.mean(df_nonindex,axis=0)*len(df)/5
    C = df_nonindex.cov()
    mu_market,risk_market = plot_efficient_frontier(m,C,0.05)
    plot_capital_line(m,C,0.05,mu_market,risk_market)
    plot_security_line(mu_market,risk_market)



print("-----------------------------------Computing Beta values------------------------------------")

def compute_beta(path, mu_market_index, risk_free_rate):
    df = pd.read_csv(path)
    df.set_index('Date', inplace=True)
    index_name = df.columns[-1]
    index_stocks = df.columns[0:10]
    nonindex_stocks = df.columns[10:20]
    beta_indices, beta_nonindices = {}, {}
    
    for i in range(len(index_stocks)):
        # Calculate covariance with market index
        cov_indices = df[index_stocks[i]].cov(df[index_name])
        cov_nonindices = df[nonindex_stocks[i]].cov(df[index_name])

        # Calculate beta
        beta_index = cov_indices / df[index_name].var()
        beta_nonindex = cov_nonindices / df[index_name].var()

        beta_indices[index_stocks[i]] = beta_index
        beta_nonindices[nonindex_stocks[i]] = beta_nonindex
    
    m_indices = np.mean(df[index_stocks], axis=0) * len(df) / 5
    m_nonindices = np.mean(df[nonindex_stocks], axis=0) * len(df) / 5

    print("\nStock Name\t\tActual Return\t\tExpected Return")
    for key in beta_indices:
        expected_return = beta_indices[key] * (mu_market_index - risk_free_rate) + risk_free_rate
        print(f"{key}\t\t{m_indices[key]}\t\t{expected_return}")
    
    for key in beta_nonindices:
        expected_return = beta_nonindices[key] * (mu_market_index - risk_free_rate) + risk_free_rate
        print(f"{key}\t\t{m_nonindices[key]}\t\t{expected_return}")
    
    return beta_indices, beta_nonindices


def q2b_2c():
    mu_market_BSE, risk_market_BSE = market_risk_and_return(BSE_PATH)
    beta_BSE, beta_nonBSE = compute_beta(BSE_PATH, mu_market_BSE, 0.05)

    mu_market_NSE,risk_market_BSE = market_risk_and_return(NSE_PATH)
    beta_NSE, beta_nonNSE = compute_beta(NSE_PATH,mu_market_NSE,0.05)

    print("---------------------------------- Beta values for BSE ------------------------------------")
    for key in beta_BSE:
        print(f"{key}\t\t=\t\t{beta_BSE[key]}")

    print("---------------------------------- Beta values for Non-BSE ------------------------------------")
    for key in beta_nonBSE:
        print(f"{key}\t\t=\t\t{beta_nonBSE[key]}")

    print("---------------------------------- Beta values for NSE ------------------------------------")
    for key in beta_NSE:
        print(f"{key}\t\t=\t\t{beta_NSE[key]}")   

    print("---------------------------------- Beta values for Non-NSE ------------------------------------")
    for key in beta_nonNSE:
        print(f"{key}\t\t=\t\t{beta_nonNSE[key]}")       


mu_market_BSE, risk_market_BSE = market_risk_and_return(BSE_PATH)
mu_market_NSE, risk_market_NSE = market_risk_and_return(NSE_PATH)
q2a(BSE_PATH,mu_market_BSE,risk_market_BSE)
q2a(NSE_PATH,mu_market_NSE,risk_market_NSE)
q2b_2c()