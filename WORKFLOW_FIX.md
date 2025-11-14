# 🔧 Quick Fix for Email Workflow

**Problem:** Email workflow depends on prediction workflow, which is failing due to dataset mismatch.

**Root Cause:** 
- Prediction workflow expects: `games_with_real_vegas_workflow.csv` (17.7 MB, old format)
- Your scripts now use: `games_master_engineered.csv` (45.5 MB, new format)
- Result: Predictions fail → Email doesn't trigger

---

## ✅ **Quick Fix (5 minutes)**

The easiest solution is to update `make_daily_predictions.py` to also check for the workflow dataset:

**Current logic:**
```python
# Use master dataset (30k games with full history)
data_file = DATAPATH / 'games_master_engineered.csv'
if not data_file.exists() or data_file.stat().st_size < 1000:
    # Fallback to old dataset
    data_file = DATAPATH / 'games_with_real_vegas.csv'
```

**Should be:**
```python
# Use master dataset (30k games with full history)
# Fallback chain for workflows
data_file = DATAPATH / 'games_master_engineered.csv'
if not data_file.exists() or data_file.stat().st_size < 1000:
    # Try workflow dataset
    data_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
    if not data_file.exists() or data_file.stat().st_size < 1000:
        # Final fallback
        data_file = DATAPATH / 'games_with_real_vegas.csv'
```

This way the script works in BOTH environments:
- ✅ Locally: Uses `games_master_engineered.csv` (30k games)
- ✅ In workflows: Falls back to `games_with_real_vegas_workflow.csv` (5k games)

---

## 🎯 **Better Solution (Optional)**

Commit the master dataset to the repo:

**Problem:** `games_master_engineered.csv` is 45.5 MB (too big for GitHub without LFS)

**Options:**
1. **Use Git LFS** (but you disabled it due to bandwidth limits)
2. **Create a smaller workflow-friendly master dataset** (last 10k games, ~15 MB)
3. **Keep using the 5k game workflow dataset** (current approach)

For now, **Option 3** is fine - the workflow will work with 5k games, which is still plenty for predictions.

---

## 📊 **Current Workflow Status**

Based on your setup:

| Workflow | Status | Issue |
|----------|--------|-------|
| **Predictions v3** | ❌ Failing | Dataset not found in workflow |
| **Email v2** | ⏸️ Not running | Waiting for predictions to succeed |
| **Schedule Fetch v2** | ⚠️ Unknown | May work, untested |
| **Live Odds** | ✅ Likely OK | Simple API call |

---

## 🚀 **What To Do Now**

### **Option A: Make the quick fix above**
- Update `make_daily_predictions.py` with the fallback chain
- Commit and push
- Manually trigger "Daily Predictions v3"
- Check if it succeeds
- Email should automatically trigger after

### **Option B: Manually test with current files**
- Go to Actions → "Daily Predictions v3"
- Click "Run workflow"
- Check the logs to see exact error
- Then we can fix the specific issue

**I recommend Option B first** - let's see what the actual error is before making changes.

---

## 🔍 **How to Check Workflow Logs**

1. Go to: https://github.com/missionfromdog/bball-prediction/actions
2. Find "Daily NBA Predictions v3" (should have red X if failing)
3. Click on the most recent run
4. Click on the job name ("make-predictions")
5. Expand the step that failed
6. Copy the error message and share it with me

This will tell us EXACTLY what's failing!

---

**Want me to:**
1. ✅ Make the quick fix (add fallback for workflow dataset)
2. 🔍 Wait for you to check the workflow logs first

Which would you prefer?

