package com.riskgrid.mining;

import java.awt.image.BufferedImage;

/**
 * Абстракція джерела зображень для клітинки території.
 * DroneAgent працює проти цього інтерфейсу, не знаючи, як саме
 * розміщені "міни" на місцевості (кластерами чи інакше) — цю логіку
 * інкапсулює SimulatedTerrain.
 */
public interface TerrainSource {

    /**
     * Повертає зображення для клітинки (i, j), яке передається в детектор,
     * або null, якщо для цієї клітинки немає окремого кадру (наприклад,
     * порожня місцевість без потенційних цілей).
     */
    BufferedImage getImageAt(int i, int j);
}
