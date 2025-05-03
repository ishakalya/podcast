import os
import sys
import time

def print_header(message):
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    print_header("EV MARKET ANALYSIS AND PREDICTION SYSTEM")
    
    print("This script will run both the basic market analysis and advanced predictive models.")
    print("\nOptions:")
    print("1. Run basic market analysis only")
    print("2. Run predictive models only")
    print("3. Run both analyses")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            print_header("RUNNING BASIC MARKET ANALYSIS")
            print("Starting basic EV market analysis...")
            import ev_market_analysis
            ev_market_analysis.main()
            print("\nBasic market analysis complete! Results saved to 'market_analysis_report.txt'")
            break
            
        elif choice == '2':
            print_header("RUNNING ADVANCED PREDICTIVE MODELS")
            print("Starting advanced predictive analysis...")
            import ev_predictive_models
            ev_predictive_models.main()
            print("\nPredictive analysis complete! Results saved to 'predictive_analysis_report.txt'")
            break
            
        elif choice == '3':
            print_header("RUNNING COMPLETE ANALYSIS SUITE")
            
            print("Step 1: Running basic market analysis...")
            import ev_market_analysis
            ev_market_analysis.main()
            print("Basic market analysis complete!")
            
            print("\nStep 2: Running advanced predictive models...")
            import ev_predictive_models
            ev_predictive_models.main()
            print("Predictive analysis complete!")
            
            print("\nAll analyses complete! Results saved to:")
            print("- 'market_analysis_report.txt'")
            print("- 'predictive_analysis_report.txt'")
            break
            
        elif choice == '4':
            print("Exiting program.")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
    
    print("\nThank you for using the EV Market Analysis and Prediction System!")

if __name__ == "__main__":
    main()