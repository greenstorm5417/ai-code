import os
from src.portfolio_train import train_portfolio
import dotenv
dotenv.load_dotenv()

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    tickers_csv = os.path.join(project_dir, "tickers.csv")

    train_results, val_result, _ = train_portfolio(
        tickers_csv=tickers_csv,
        max_symbols=25,  # Portfolio of up to 25 stocks
        window_size=60,  # minutes of history per observation
        initial_cash=10000.0,  # Starting capital
        n_episodes=3,  # Train for 3 episodes
        target_days=90,  # Days of historical data (Alpha Vantage: 90 days)
        seed=42,
    )

    print("\n=== Training Summary ===")
    for r in train_results:
        print(f"Episode {r['episode']}: {r['pct_return']:.2f}% return, "
              f"final=${r['final_value']:.2f}, avg_loss={r['avg_loss']:.4f}")
    
    print("\n=== Validation Summary ===")
    print(f"Return: {val_result['pct_return']:.2f}%")
    print(f"Final value: ${val_result['final_value']:.2f}")


if __name__ == "__main__":
    main()
