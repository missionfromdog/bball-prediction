# Available Sportsbooks from The Odds API

This document lists all sportsbooks available through The Odds API for NBA betting odds.

## Major US Sportsbooks

| Sportsbook | API Key | Display Name | Status | Notes |
|------------|---------|--------------|--------|-------|
| DraftKings | `draftkings` | DraftKings | ✅ Active | One of the largest US sportsbooks |
| FanDuel | `fanduel` | FanDuel | ✅ Active | Major competitor to DraftKings |
| BetMGM | `betmgm` | BetMGM | ✅ Active | MGM Resorts sportsbook |
| Caesars | `caesars` | Caesars Sportsbook | ✅ Active | Caesars Entertainment sportsbook |
| BetRivers | `betrivers` | BetRivers | ✅ Active | Rush Street Interactive |
| PointsBet | `pointsbet` | PointsBet | ✅ Active | Australian-based, US operations |
| WynnBET | `wynnbet` | WynnBET | ✅ Active | Wynn Resorts sportsbook |
| Barstool Sportsbook | `barstool` | Barstool Sportsbook | ✅ Active | Penn National Gaming |
| Fox Bet | `foxbet` | FOX Bet | ✅ Active | FOX Sports betting platform |
| Unibet | `unibet_us` | Unibet | ✅ Active | Kindred Group (US operations) |
| BetOnline | `betonlineag` | BetOnline.ag | ✅ Active | Offshore, accepts US players |
| Bovada | `bovada` | Bovada | ✅ Active | Offshore, accepts US players |
| MyBookie | `mybookie` | MyBookie.ag | ✅ Active | Offshore, accepts US players |
| SportsBetting | `sportsbetting` | SportsBetting.ag | ✅ Active | Offshore, accepts US players |

## Additional Sportsbooks (May Vary by Region)

| Sportsbook | API Key | Display Name | Status | Notes |
|------------|---------|--------------|--------|-------|
| Hard Rock Bet | `hardrockbet` | Hard Rock Bet | ⚠️ Varies | Regional availability |
| Betfred | `betfred` | Betfred Sports | ⚠️ Varies | Regional availability |
| Bet365 | `bet365` | Bet365 | ⚠️ Varies | Limited US availability |
| ESPN BET | `espnbet` | ESPN BET | ⚠️ Varies | Newer platform |
| Circa Sports | `circasports` | Circa Sports | ⚠️ Varies | Regional (Nevada, Colorado) |
| SuperBook | `superbook` | SuperBook Sports | ⚠️ Varies | Regional availability |
| Betway | `betway` | Betway | ⚠️ Varies | Limited US availability |

## Currently Filtered in Code

The following sportsbooks are currently included in the filter (Major US + Bet365, Hard Rock Bet, ESPN BET):

### Major US Sportsbooks (10)
1. **DraftKings** - `draftkings`
2. **FanDuel** - `fanduel`
3. **BetMGM** - `betmgm`
4. **Caesars** - `caesars`
5. **BetRivers** - `betrivers`
6. **PointsBet** - `pointsbet`
7. **WynnBET** - `wynnbet`
8. **Barstool** - `barstool`
9. **FOX Bet** - `foxbet`
10. **Unibet** - `unibet_us`

### Additional Major Books (3)
11. **Bet365** - `bet365`
12. **Hard Rock Bet** - `hardrockbet`
13. **ESPN BET** - `espnbet`

### Regional (Optional - 3)
14. **Circa Sports** - `circasports`
15. **SuperBook** - `superbook`
16. **Betway** - `betway`

### Excluded (Offshore Books)
- ❌ **Bovada** - `bovada` (Offshore)
- ❌ **MyBookie** - `mybookie` (Offshore)
- ❌ **BetOnline** - `betonlineag` (Offshore)
- ❌ **SportsBetting** - `sportsbetting` (Offshore)

## Currently Found in Data

Based on recent API responses, the following sportsbooks have been found:

1. **DraftKings** - Most common
2. **FanDuel** - Most common
3. **BetMGM** - Regular
4. **BetRivers** - Regular
5. **BetOnline.ag** - Occasional
6. **Bovada** - Occasional
7. **MyBookie.ag** - Occasional

## Notes

- **Status**: ✅ Active = Regularly available, ⚠️ Varies = Availability depends on region/date
- **Offshore books** (BetOnline, Bovada, MyBookie, SportsBetting) accept US players but may not be licensed in all states
- **Regional books** may only appear in specific states
- The Odds API may add or remove sportsbooks over time
- Some sportsbooks may have multiple API keys (e.g., `draftkings`, `draftkings_sportsbook`)

## Recommendation

For a comprehensive comparison, consider including:
- **Top 5-7 major books**: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, PointsBet, WynnBET
- **Regional favorites**: Based on your location
- **Offshore options**: If you want broader coverage (BetOnline, Bovada)

This gives you a good mix of major operators, regional options, and offshore alternatives for finding the best odds.

