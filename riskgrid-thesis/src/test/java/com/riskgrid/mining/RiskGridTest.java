package com.riskgrid.mining;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

class RiskGridTest {

    private RiskGrid grid;

    @BeforeEach
    void setUp() {
        grid = new RiskGrid(10, 10, 1.0, 0.0, 0.0);
    }

    // --- risk/uncertainty завжди в межах (0,1) ---

    @Test
    void riskStaysWithinBoundsAfterUpdate() {
        grid.update(2, 3, 0.9, 0.8, 1000L);
        double r = grid.getRisk(2, 3);
        assertTrue(r > 0.0 && r < 1.0, "risk має бути в (0,1), отримано " + r);
    }

    @Test
    void uncertaintyStaysWithinBoundsAfterUpdate() {
        grid.update(2, 3, 0.9, 0.8, 1000L);
        double u = grid.getUncertainty(2, 3);
        assertTrue(u > 0.0 && u < 1.0, "uncertainty має бути в (0,1), отримано " + u);
    }

    @Test
    void boundsHoldAfterManyRepeatedUpdates() {
        for (int k = 0; k < 1000; k++) {
            double z = (k % 2 == 0) ? 0.99 : 0.01; // навмисно суперечливі спостереження
            grid.update(5, 5, z, 1.0, k);
            double r = grid.getRisk(5, 5);
            double u = grid.getUncertainty(5, 5);
            assertTrue(r > 0.0 && r < 1.0, "risk вийшов за межі на ітерації " + k + ": " + r);
            assertTrue(u > 0.0 && u < 1.0, "uncertainty вийшла за межі на ітерації " + k + ": " + u);
        }
    }

    // --- n і uncertainty монотонні ---

    @Test
    void observationCountNeverDecreases() {
        assertEquals(0L, grid.getObservationCount(4, 4));
        grid.update(4, 4, 0.5, 0.5, 1L);
        assertEquals(1L, grid.getObservationCount(4, 4));
        grid.update(4, 4, 0.7, 0.3, 2L);
        assertEquals(2L, grid.getObservationCount(4, 4));
    }

    @Test
    void uncertaintyMonotonicallyDecreasesRegardlessOfObservationOutcome() {
        double u0 = grid.getUncertainty(3, 3); // жодного спостереження -> максимальна
        grid.update(3, 3, 0.95, 0.5, 1L);
        double u1 = grid.getUncertainty(3, 3);
        grid.update(3, 3, 0.05, 0.5, 2L); // суперечливе спостереження
        double u2 = grid.getUncertainty(3, 3);

        assertTrue(u1 < u0, "uncertainty має спадати після 1-го спостереження");
        assertTrue(u2 < u1, "uncertainty має спадати навіть при суперечливому спостереженні");
    }

    // --- незалежність клітинок ---

    @Test
    void updatingOneCellDoesNotAffectAnother() {
        double riskBefore = grid.getRisk(7, 7);
        grid.update(1, 1, 0.95, 0.9, 500L);
        assertEquals(riskBefore, grid.getRisk(7, 7), 1e-9);
    }

    // --- ключовий інваріант 2.7: адитивність, незалежність від порядку ---

    @Test
    void updateOrderDoesNotAffectFinalRisk() {
        RiskGrid gridA = new RiskGrid(5, 5, 1.0, 0.0, 0.0);
        RiskGrid gridB = new RiskGrid(5, 5, 1.0, 0.0, 0.0);

        gridA.update(2, 2, 0.8, 0.5, 1L);
        gridA.update(2, 2, 0.3, 0.6, 2L);
        gridA.update(2, 2, 0.6, 0.4, 3L);

        gridB.update(2, 2, 0.6, 0.4, 3L);
        gridB.update(2, 2, 0.8, 0.5, 1L);
        gridB.update(2, 2, 0.3, 0.6, 2L);

        // Допуск на похибку через порядок операцій з плаваючою комою (не логічну помилку) -
        // додавання double в різному порядку може давати мікроскопічні відхилення в останніх
        // розрядах (IEEE-754 не строго асоціативна для округлення), тому порівнюємо з epsilon.
        assertEquals(gridA.getRisk(2, 2), gridB.getRisk(2, 2), 1e-9);
        assertEquals(gridA.getUncertainty(2, 2), gridB.getUncertainty(2, 2), 1e-9);
    }

    // --- координати ---

    @Test
    void coordinateMappingIsCorrect() {
        RiskGrid g = new RiskGrid(10, 10, 2.0, 100.0, 50.0);
        int[] idx = g.toCellIndex(104.5, 53.0);
        assertEquals(2, idx[0]);
        assertEquals(1, idx[1]);
    }

    // --- валідація ---

    @Test
    void updateThrowsForOutOfBoundsCell() {
        assertThrows(IllegalArgumentException.class, () -> grid.update(100, 100, 0.5, 0.5, 1L));
    }

    @Test
    void updateThrowsForZAtBoundary() {
        assertThrows(IllegalArgumentException.class, () -> grid.update(0, 0, 1.0, 0.5, 1L));
        assertThrows(IllegalArgumentException.class, () -> grid.update(0, 0, 0.0, 0.5, 1L));
    }

    @Test
    void updateThrowsForInvalidW() {
        assertThrows(IllegalArgumentException.class, () -> grid.update(0, 0, 0.5, 0.0, 1L));
    }
}
