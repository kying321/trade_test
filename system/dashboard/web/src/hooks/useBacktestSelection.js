import { useEffect, useMemo, useState } from "react";

export function useBacktestSelection(backtests = []) {
  const [selectedBacktestId, setSelectedBacktestId] = useState("");

  useEffect(() => {
    if (!selectedBacktestId && backtests.length) setSelectedBacktestId(backtests[0].id);
  }, [backtests, selectedBacktestId]);

  const selectedBacktest = useMemo(
    () => backtests.find((item) => item.id === selectedBacktestId) || backtests[0],
    [backtests, selectedBacktestId]
  );

  return {
    selectedBacktest,
    selectedBacktestId,
    setSelectedBacktestId,
  };
}
