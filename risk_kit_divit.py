import numpy as np
import pandas as pd
import math
import matplotlib as plt
import scipy as sp
from scipy.optimize import minimize

def skewness(r):
    new_mean=0
    for i in r:
        new_mean+=((i-np.mean(r))**3)/len(r)
    s=new_mean/((np.std(r))**3)
    return s

def kurtosis(r):
    new_mean=0
    for i in r:
        new_mean+=((i-np.mean(r))**4)/len(r)
    k=new_mean/((np.std(r))**4)
    return k

def annualize_returns(r, time):
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(time/n_periods)-1

def annualize_vol(r, time):
    return r.std()*(time**0.5)

def sharpe(r,rfr,time):
    return ((np.mean(r)-rfr)/annualize_vol(r,time))

def jb_test(r):
    s=skewness(r)
    k=kurtosis(r)
    jb=(len(r)/6)*(s**2+((k-3)**2)/4)
    if jb==0:
        return True
    else:
        return False
    
def wealth_index(r):
    return (1+r).cumprod()

def peak(r):
    return r.cummax()

def drawdown(r):
    previous_peaks=wealth_index(r).cummax()
    return (wealth_index(r)-previous_peaks)/previous_peaks

def semi_deviation(r):
    new_mean=0
    for i in r:
        if i<np.average(r):
            new_mean+=(np.average(r)-i)**2
    s=new_mean/len(r)
    return math.sqrt(s)

def cutoff_value(r,alpha):
    new_arr=np.sort(r)
    return (new_arr[int(alpha*0.01*len(r))])

def hist_var(r):
    return np.average(r)-cutoff_value(r,5)
    
def hist_cvar(r):
    sum=0
    count=0
    for i in np.sort(r):
        if i<cutoff_value(r,5):
            sum+=i
            count+=1
    return (sum/count)

def gaussian_var(r):
    return -(np.average(r)+np.std(r)*1.645)

def gaussian_cvar(r):
    sum=0
    count=0
    for i in np.sort(r):
        if i<gaussian_var(r):
            sum+=i
            count+=1
    return (sum/count)

def cf_var(r,z=-1.645):
    ans=z+(z**2-1)*skewness(r)/6+(z**3-3*z)*kurtosis(r)/24-(2*(z**3)-5*z)*((skewness(r))**2)/36
    return  -(np.average(r)+np.std(r)*ans)

def portfolio_return(weights, returns):
    return weights.T@returns

def portfolio_vol(weights, cov):
    return weights.T@cov@weights

def minimize_vol(target_return, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return -portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    """
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov) 
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-')


def msr(riskfree_rate, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def portfolio_sharpe(portf_return,rfr,portf_vol):
    return (portf_return-rfr)/portf_vol

def cppi(risky_r,safe_r=None,m=3,start=100000,floor=0.85,rfr=0.03,drawdown=None):
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=floor*start
    
    if isinstance(risky_r,pd.Series):
        risky_r=pd.DataFrame(risky_r,columns=["R"])
        
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=rfr/12
    
    account_history=pd.DataFrame().reindex_like(risky_r)
    floor_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)
    peak_history=pd.DataFrame().reindex_like(risky_r)
    peak=0
    for step in range(len(dates)):
        cushion=(account_value-floor_value)/account_value
        risky_w=m*cushion
        risky_w=np.minimum(risky_w,1)
        risky_w=np.maximum(risky_w,0)
        safe_w=1-risky_w
        risky_alloc=account_value*risky_w
        safe_alloc=account_value*safe_w
        account_value=risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value=peak*(1-drawdown)
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
        floor_history.iloc[step]=floor_value
        peak_history.iloc[step]=peak
            
            
    risky_wealth=start*(1+risky_r).cumprod()
    
    
    
    backtest={
        "Wealth":account_history,
        "Risky Wealth":risky_wealth,
        "Risk Budget":cushion_history,
        "Risky Allocation":risky_w_history,
        "Multiplier":m,
        "Start":start,
        "Floor":floor,
        "Risky Asset Returns":risky_r,
        "Safe Asset Returns":safe_r,
        "Peak Value":peak_history,
        "Drawdown":drawdown,
        "Floor History":floor_history
    }
    return backtest


## def summary_stats(r,rfr=0.03):
##     ann_r=rk.annualize_returns(r,12)
##     ann_vol=rk.annualize_vol(r,12)
##     ann_sr=rk.sharpe(r,rfr,12)
##     dd=max(rk.drawdown(r))
##     sk=rk.skewness(r)
##     kurt=rk.kurtosis(r)
##     cfvar=rk.cf_var(r)
##     peak=max(rk.peak(r))
##     hist_var=rk.hist_var(r)
##     return pd.DataFrame({
##         "Annualized Returns":ann_r,
##         "Annualized Volatility":ann_vol,
##         "Sharpe Ratio":ann_sr,
##         "Skewness":sk,
##         "Kurtosis":kurt,
##         "Cornish-Fisher Var(5%)":cfvar,
##         "Historical Var(5%)":hist_var,
##         "Peak":peak,
##         "Max Drawdown":dd
##     })


    