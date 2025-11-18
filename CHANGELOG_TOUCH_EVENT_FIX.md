# 터치 이벤트 충돌 해결 변경 사항

## 문제 상황
10% 카드 상태에서 음식점 마커를 클릭하고, 50% 카드가 올라오는 애니메이션 중에 카드의 가게 제목 부분을 잡으면 지도와 카드가 같이 움직이는 현상 발생

## 원인 분석
- `.sheet-header-common` 영역에 터치 제어가 없어 기본 동작(pan-x, pan-y 모두)이 허용됨
- 애니메이션 중 헤더를 터치하면 이벤트가 지도로 전파되어 지도가 움직임
- `touch-action`은 상속되지 않으므로 자식 요소에도 명시적으로 설정해야 함

## 해결 방법
CSS와 JavaScript를 조합하여 이중 방어:
1. CSS: `touch-action: none`으로 모든 터치 동작 차단
2. JavaScript: `stopPropagation()`으로 이벤트 전파 차단

## 변경된 파일

### 1. `frontend/src/components/Map/Map.css`

**변경 위치**: `.sheet-header-common` 스타일 (223-233줄)

**변경 전**:
```css
.sheet-header-common {
  padding: 20px 20px 16px 20px;
  flex-shrink: 0;
}

.sheet-header-common h2 {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 8px 0;
}
```

**변경 후**:
```css
.sheet-header-common {
  padding: 20px 20px 16px 20px;
  flex-shrink: 0;
  touch-action: none; /* 모든 터치 동작 차단하여 지도 움직임 방지 */
  -webkit-user-drag: none;
  user-select: none;
}

.sheet-header-common h2 {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 8px 0;
  touch-action: none; /* 모든 터치 동작 차단하여 지도 움직임 방지 */
}
```

### 2. `frontend/src/components/Map/MapBottomSheet.js`

**변경 위치**: 헤더 div 요소 (292줄)

**변경 전**:
```javascript
<div className="sheet-header-common">
  <h2>{selectedRestaurant.name}</h2>
  <div className="ai-scores">
    ...
  </div>
</div>
```

**변경 후**:
```javascript
<div 
  className="sheet-header-common"
  onTouchStart={(e) => e.stopPropagation()}
  onTouchMove={(e) => e.stopPropagation()}
  onTouchEnd={(e) => e.stopPropagation()}
  onTouchCancel={(e) => e.stopPropagation()}
>
  <h2>{selectedRestaurant.name}</h2>
  <div className="ai-scores">
    ...
  </div>
</div>
```

## 되돌리기 방법

### 방법 1: Git을 사용하는 경우
```bash
git checkout HEAD -- frontend/src/components/Map/Map.css frontend/src/components/Map/MapBottomSheet.js
```

### 방법 2: 수동으로 되돌리기

**Map.css 되돌리기**:
- `.sheet-header-common`에서 다음 속성 제거:
  - `touch-action: none;`
  - `-webkit-user-drag: none;`
  - `user-select: none;`
- `.sheet-header-common h2`에서 다음 속성 제거:
  - `touch-action: none;`

**MapBottomSheet.js 되돌리기**:
- 헤더 div에서 다음 속성 제거:
  - `onTouchStart={(e) => e.stopPropagation()}`
  - `onTouchMove={(e) => e.stopPropagation()}`
  - `onTouchEnd={(e) => e.stopPropagation()}`
  - `onTouchCancel={(e) => e.stopPropagation()}`

## 기존 코드 유지 사항

다음 CSS는 **그대로 유지**해야 합니다 (제거하지 마세요):

```css
/* 바텀시트 터치 이벤트 제어 - 세로 드래그만 허용하여 지도 움직임 방지 */
[data-rsbs-overlay],
[data-rsbs-backdrop],
[data-rsbs-header],
[data-rsbs-scroll] {
  touch-action: pan-y !important;
  -webkit-user-drag: none;
  user-select: none;
}

.bottom-sheet-content {
  background: white;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  touch-action: pan-y !important;
}
```

이 부분은 라이브러리 요소들과 다른 콘텐츠 영역에서 정상적으로 작동하므로 유지해야 합니다.

## 테스트 방법

1. 10% 카드 상태에서 음식점 마커 클릭
2. 50% 카드가 올라오는 애니메이션 중에 가게 제목 부분을 잡아서 드래그
3. 지도가 움직이지 않고 카드만 움직이는지 확인

## 변경 일시
2024년 (현재 날짜)

