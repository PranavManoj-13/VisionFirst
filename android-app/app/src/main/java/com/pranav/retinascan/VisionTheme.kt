package com.pranav.retinascan

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily

internal val Orange = Color(0xFFBD4100)
internal val Black = Color(0xFF151515)
internal val Grey = Color(0xFF5E5E5E)
internal val Line = Color(0xFFE5E2DF)
internal val Paper = Color(0xFFFAF9F7)
@Composable
internal fun VisionTheme(content: @Composable () -> Unit) {
    val base = Typography()
    MaterialTheme(
        colorScheme = lightColorScheme(primary=Orange,onPrimary=Color.White,background=Color.White,
            surface=Color.White,onSurface=Black,onBackground=Black,onSurfaceVariant=Grey,
            surfaceVariant=Paper,secondary=Black,outline=Line),
        typography=Typography(
            headlineLarge=base.headlineLarge.copy(fontFamily=FontFamily.SansSerif),
            headlineMedium=base.headlineMedium.copy(fontFamily=FontFamily.SansSerif),
            titleLarge=base.titleLarge.copy(fontFamily=FontFamily.SansSerif),
            bodyLarge=base.bodyLarge.copy(fontFamily=FontFamily.SansSerif),
            bodyMedium=base.bodyMedium.copy(fontFamily=FontFamily.SansSerif)),content=content)
}
