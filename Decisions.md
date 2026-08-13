Decisions.md

Collection:
    Three separate connections: \public and \market futures separately
    Wanted to avoid limited data flow if the connections were unrouted

    
    Convert to JSONL first then to_parquet handles conversion
    Avoiding bad conversions being irrepairable (I only want to download once but might need to convert multiple times)
    
    gzip file for each hour: torn members can exist but exposes at most an hour of data to being in a corrupted file

    Chose heartbeat giving indication for each minute and each stream rather than for each connection

    timestamps and wall-clock for each record

    Laptop for collection due to familiarity
    However, this meant two long gaps before power settings fixed

Data:
    recorded 100+hrs of data with large gaps
    
    Chose last 80 hrs as near continuous with only small minor gaps (excludes longer gaps in data)
    
    dropped data whose price was stale for more than 2s to avoid inventing data

    198 data points lost for stale quotes (0.07%). Mostly single points, likely a genuinely quiet book rather than connection loss
    
Target:
    Spot was target but used futures as features

    Rejected resample('1s').last(): the bin [t, t+1s) is labelled t but contains observations up to t+0.999s
    
    treated dataframe as a grid with each second corresponds to each second
    Created complete grid with no missing seconds: 5s shift corresponds to 5 rows

Baseline Features:
    Chose 4 different groups of Baseline features:
        Book:
            spot_imbalance, fut_imbalance: These correspond to relationship between bid and ask quantities
            spot_spread_bps, fut_spread_bps: spreads between ask and bid
            spot_wmid_dev_bps, fut_wmid_dev_bps: mid weighted by size of positions minus mid
        between venue:
            basis_bps: relationship between the mid-price of futures and spot
        Trade flow:
            _ofi_1s, _ofi_5s, fut_ofi_60s: signed flow normalised by volume over different timescales (implicitly gives momentum features for volume)
        Volume/momentum
            rv_10s_bps, rv_60s_bps: rolling standard deviations (longer timescales as volatility is slower-varying)
            ret_1s_bps: previous 1 second log-returns

Baseline:
    Baseline chosen was simple ridge for interpretability and simplicity (can test whether additional complexity helps)

    Chose walk-forward for cross validation that works for time-series (trains on past test on future)

    60s embargo between each fold and also the holdout (60s is the length of longest feature window)
        

Feature Engineering:
    Considered time of day but little consistent correlation between each day, insufficient number of days to have enough datapoints for consideration
    day-to-day correlations 0.12 / 0.22 / −0.32
    
    Analysed different interactions based on which features correlated together
    Imbalance vs spread: high spread and more bids might mean greater change in price
    Imbalance vs volatility: similar to above

    None of these showed significant interactive power besides linear combinations (imb_diff=fut_imbalance − spot_imbalance exact linear combination not used for ridge but showed no improvement over baseline).

    Looked at lags over 1,2, and 5 seconds. Did not improve over baseline so ignored
    +0.0006 to ΔR²_zero vs fold σ 0.024

Modelling:
    Compared lightgbm against existing ridge and it provided no improvement over baseline
    +0.019 to ΔR²_zero vs fold σ 0.036, with higher variance and worse DA
    
