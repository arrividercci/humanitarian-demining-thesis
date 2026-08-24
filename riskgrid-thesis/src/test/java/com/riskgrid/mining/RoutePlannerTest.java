package com.riskgrid.mining;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

class RoutePlannerTest {

    @Test
    void selectsCellWithMaximumUtility() {
        RiskGrid grid = new RiskGrid(5, 5, 1.0, 0.0, 0.0);
        RoutePlanner planner = new RoutePlanner(grid, 1.0, 0.0, 0.0);

        grid.update(1, 1, 0.9, 0.8, 1L);
        grid.update(3, 2, 0.6, 0.3, 2L);
        grid.update(0, 4, 0.95, 0.9, 3L);
        grid.update(2, 2, 0.4, 0.5, 4L);
        grid.update(0, 4, 0.85, 0.6, 5L);

        RoutePlanner.Point p = new RoutePlanner.Point(2.5, 2.5);
        double lambda = 0.1;
        double radius = 10.0;

        RoutePlanner.TargetResult chosen = planner.selectNextTarget(p, radius, lambda);

        // Незалежно рахуємо корисність brute-force для всіх клітинок карти
        // тими самими формулами (H(risk)*unc, розділ 3.3) і шукаємо максимум
        double bestUtility = Double.NEGATIVE_INFINITY;
        int bestI = -1, bestJ = -1;
        for (int i = 0; i < grid.getRows(); i++) {
            for (int j = 0; j < grid.getCols(); j++) {
                double centerX = 0.0 + (i + 0.5) * 1.0;
                double centerY = 0.0 + (j + 0.5) * 1.0;
                double dist = Math.sqrt(Math.pow(centerX - p.x(), 2) + Math.pow(centerY - p.y(), 2));
                double risk = grid.getRisk(i, j);
                double h = (risk > 0.0 && risk < 1.0) ? -risk * Math.log(risk) - (1 - risk) * Math.log(1 - risk) : 0.0;
                double u = h * grid.getUncertainty(i, j) - lambda * dist;
                if (u > bestUtility) {
                    bestUtility = u;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        assertEquals(bestI, chosen.i(), "планувальник обрав не той рядок клітинки з максимальною корисністю");
        assertEquals(bestJ, chosen.j(), "планувальник обрав не той стовпець клітинки з максимальною корисністю");
    }

    @Test
    void higherLambdaPrefersCloserCellsOverFartherOnes() {
        // Усі клітинки в радіусі отримують ІДЕНТИЧНЕ спостереження -> однаковий
        // інформаційний виграш скрізь. Єдина відмінність, що лишається -
        // відстань, тож планувальник повинен обрати найближчу клітинку до p.
        RiskGrid grid = new RiskGrid(10, 1, 1.0, 0.0, 0.0);
        for (int i = 0; i < 10; i++) {
            grid.update(i, 0, 0.8, 0.5, 1L);
        }

        RoutePlanner planner = new RoutePlanner(grid, 1.0, 0.0, 0.0);
        RoutePlanner.Point p = new RoutePlanner.Point(0.5, 0.5); // найближче до клітинки (0,0)

        RoutePlanner.TargetResult chosen = planner.selectNextTarget(p, 20.0, /* lambda = */ 1.0);

        assertEquals(0, chosen.i(), "при однаковому інформаційному виграші має бути обрана найближча клітинка");
    }

    //Розширення радіусу, коли поруч немає кандидатів ---

    @Test
    void expandsRadiusWhenNoCandidatesInInitialRadius() {
        RiskGrid grid = new RiskGrid(20, 20, 1.0, 0.0, 0.0);
        RoutePlanner planner = new RoutePlanner(grid, 1.0, 0.0, 0.0);

        RoutePlanner.Point p = new RoutePlanner.Point(0.0, 0.0);
        double tooSmallRadius = 0.1;

        RoutePlanner.TargetResult chosen =
                assertDoesNotThrow(() -> planner.selectNextTarget(p, tooSmallRadius, 0.1));

        assertTrue(chosen.i() >= 0 && chosen.j() >= 0, "має бути знайдена коректна клітинка після розширення радіусу");
    }

    //Тест формули 3.5 складність не залежить від розміру карти

    @Test
    void candidateCountIsIndependentOfMapSize() {
        double cellSize = 1.0;
        double radius = 10.0;

        RiskGrid smallGrid = new RiskGrid(100, 100, cellSize, 0.0, 0.0);
        RoutePlanner smallPlanner = new RoutePlanner(smallGrid, cellSize, 0.0, 0.0);
        RoutePlanner.TargetResult smallResult =
                smallPlanner.selectNextTarget(new RoutePlanner.Point(50, 50), radius, 0.1);
        int candidatesSmall = smallResult.candidatesEvaluated();

        RiskGrid bigGrid = new RiskGrid(1000, 1000, cellSize, 0.0, 0.0);
        RoutePlanner bigPlanner = new RoutePlanner(bigGrid, cellSize, 0.0, 0.0);
        RoutePlanner.TargetResult bigResult =
                bigPlanner.selectNextTarget(new RoutePlanner.Point(500, 500), radius, 0.1);
        int candidatesBig = bigResult.candidatesEvaluated();

        assertEquals(candidatesSmall, candidatesBig,
                "кількість розглянутих кандидатів має бути однаковою незалежно від розміру карти (100x100 проти 1000x1000)");

        assertTrue(candidatesSmall > 250 && candidatesSmall < 400,
                "кількість кандидатів має відповідати площі кола радіуса R, отримано " + candidatesSmall);
    }

    //Виключення застовплених клітинок (координація дронів, розділ 3.6)

    @Test
    void excludedCellsAreNeverChosen() {
        RiskGrid grid = new RiskGrid(5, 5, 1.0, 0.0, 0.0);
        RoutePlanner planner = new RoutePlanner(grid, 1.0, 0.0, 0.0);

        // Найпривабливіша клітинка -- (2,2): найвищий ризик і невизначеність
        grid.update(2, 2, 0.95, 0.9, 1L);
        grid.update(1, 1, 0.5, 0.4, 2L);

        RoutePlanner.Point p = new RoutePlanner.Point(2.5, 2.5);

        // Без виключень планувальник обирає (2,2)
        RoutePlanner.TargetResult unrestricted = planner.selectNextTarget(p, 10.0, 0.1);
        assertEquals(2, unrestricted.i());
        assertEquals(2, unrestricted.j());

        // З виключеною (2,2) планувальник має обрати іншу клітинку
        Set<Long> excluded = Set.of(DroneCoordinator.cellKey(2, 2));
        RoutePlanner.TargetResult restricted = planner.selectNextTarget(p, 10.0, 0.1, excluded);

        assertFalse(restricted.i() == 2 && restricted.j() == 2,
                "виключена клітинка не повинна бути обрана повторно");
    }

    @Test
    void emptyExclusionSetBehavesLikeNoExclusion() {
        RiskGrid grid = new RiskGrid(5, 5, 1.0, 0.0, 0.0);
        RoutePlanner planner = new RoutePlanner(grid, 1.0, 0.0, 0.0);
        grid.update(2, 2, 0.9, 0.8, 1L);

        RoutePlanner.Point p = new RoutePlanner.Point(2.5, 2.5);

        RoutePlanner.TargetResult withEmptySet =
                planner.selectNextTarget(p, 10.0, 0.1, Collections.emptySet());
        RoutePlanner.TargetResult withoutSet =
                planner.selectNextTarget(p, 10.0, 0.1);

        assertEquals(withoutSet.i(), withEmptySet.i());
        assertEquals(withoutSet.j(), withEmptySet.j());
    }

    // --- Потоко-безпечність: конкурентні виклики одного спільного RoutePlanner ---
    // Доводить, що прибране мутабельне поле instance-рівня справді усунуло
    // гонку даних: багато потоків одночасно викликають selectNextTarget на
    // ОДНОМУ екземплярі планувальника, і кожен виклик повертає коректний,
    // узгоджений з brute-force результат -- без пошкодження внутрішнього стану.
    @Test
    void concurrentCallsOnSharedPlannerAreSafe() throws InterruptedException {
        RiskGrid grid = new RiskGrid(30, 30, 1.0, 0.0, 0.0);
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
                grid.update(i, j, 0.3 + 0.01 * ((i + j) % 7), 0.5, 1L);
            }
        }
        RoutePlanner planner = new RoutePlanner(grid, 1.0, 0.0, 0.0);
        RoutePlanner.Point p = new RoutePlanner.Point(15.0, 15.0);

        int threadCount = 16;
        ExecutorService pool = Executors.newFixedThreadPool(threadCount);
        AtomicInteger mismatches = new AtomicInteger(0);

        // Очікуваний результат обчислюємо один раз послідовно заздалегідь
        RoutePlanner.TargetResult expected = planner.selectNextTarget(p, 10.0, 0.1);

        for (int t = 0; t < threadCount; t++) {
            pool.submit(() -> {
                RoutePlanner.TargetResult r = planner.selectNextTarget(p, 10.0, 0.1);
                if (r.i() != expected.i() || r.j() != expected.j()
                        || r.candidatesEvaluated() != expected.candidatesEvaluated()) {
                    mismatches.incrementAndGet();
                }
            });
        }

        pool.shutdown();
        assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS), "потоки не завершились вчасно");
        assertEquals(0, mismatches.get(), "конкурентні виклики на спільному планувальнику дали розбіжні результати");
    }

    // DroneCoordinator: жодні два дрони не застовплюють ту саму клітинку

    @Test
    void coordinatorNeverAllowsDoubleClaimUnderConcurrency() throws InterruptedException {
        DroneCoordinator coordinator = new DroneCoordinator();
        int attempts = 500;
        ExecutorService pool = Executors.newFixedThreadPool(20);
        AtomicInteger successfulClaims = new AtomicInteger(0);

        for (int k = 0; k < attempts; k++) {
            pool.submit(() -> {
                if (coordinator.tryClaim(3, 3)) {
                    successfulClaims.incrementAndGet();
                }
            });
        }

        pool.shutdown();
        assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS), "потоки не завершились вчасно");

        assertEquals(1, successfulClaims.get(),
                "рівно одна спроба застовпити клітинку (3,3) має вдатись серед " + attempts + " конкурентних спроб");
    }
}