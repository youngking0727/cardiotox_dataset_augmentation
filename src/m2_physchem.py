"""M2: 理化性质计算器（兼容入口，实现见 m2_physchem_calculator）。"""

from m2_physchem_calculator import PhysChemCalculator, create_physchem_calculator, _cli_main

__all__ = ["PhysChemCalculator", "create_physchem_calculator"]

if __name__ == "__main__":
    _cli_main()
